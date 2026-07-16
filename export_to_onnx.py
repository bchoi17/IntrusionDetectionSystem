"""
export_to_onnx.py

This script converts the two models used in the two-stage IDS model into one combined ONNX model so the model is ready
for deployment to the Triton Inference Server.

Just like the two-stage IDS model, the ONNX model first predicts whether each network flow is normal or an attack, and
the rows that are predicted as attacks then receive an attack-type prediction.

The imbalanced-learn samplers in the saved pipelines are training-only steps.
They are deliberately removed before conversion; inference uses the fitted
StandardScaler and RandomForestClassifier from each pipeline.
"""

from pathlib import Path
from typing import Tuple
import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnx import TensorProto, compose, helper
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline

# Resolve every file location relative to this script so the exporter works regardless of the terminal's working
# directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Locations of the trained scikit-learn models and generated riton repository
MODEL_DIR = PROJECT_ROOT / "models"
TRITON_MODEL_DIR = PROJECT_ROOT / "triton_model_repository" / "two_stage_ids"
ONNX_PATH = TRITON_MODEL_DIR / "1" / "model.onnx"
TRITON_CONFIG_PATH = TRITON_MODEL_DIR / "config.pbtxt"

# ONNX operator-set version
TARGET_OPSET = 17

# Return only the fitted steps that execute during inference
def inference_pipeline(fitted_pipeline) -> Pipeline:
    # Both scalar and classifier must exist the scalar prepares the features while the classifier does the prediciton
    required_steps = ("scaler", "classifier")

    # Check the model before converting to ONNX model
    missing = [name for name in required_steps if name not in fitted_pipeline.named_steps]
    if missing:
        raise ValueError(f"Pipeline is missing required steps: {missing}")

    # Reuse the already fitted scaler and classifier since the model does not need to be retrained
    return Pipeline(
        [
            ("scaler", fitted_pipeline.named_steps["scaler"]),
            ("classifier", fitted_pipeline.named_steps["classifier"]),
        ]
    )

# Convert one scaler/classifier pair into an ONNX model
def convert_stage(fitted_pipeline, input_name: str) -> onnx.ModelProto:
    # Count number of features when original model was fitted
    number_of_features = int(fitted_pipeline.n_features_in_)

    # Remove training-only sampling stages before conversion
    pipeline = inference_pipeline(fitted_pipeline)

    # Convert the inference model. None represents a dynamic batch size
    return convert_sklearn(
        pipeline,
        initial_types=[(input_name, FloatTensorType([None, number_of_features]))],
        options={id(pipeline): {"zipmap": False}},
        target_opset=TARGET_OPSET,
    )

# Namespace a stage graph and reconnect it to the shared IDS input
def prefix_stage(model: onnx.ModelProto, prefix: str, old_input: str) -> onnx.ModelProto:
    # Namespace every graph element such as binary_label and attack_label
    model = compose.add_prefix(model, prefix)

    # Determine the input name after the prefix is added
    prefixed_input = f"{prefix}{old_input}"

    # Replace references to the stage-specific input with the shared input used by the final model
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == prefixed_input:
                node.input[index] = "network_features"

    # Remove the stage's separate graph input
    del model.graph.input[:]
    return model

# Compose both classifiers and apply the two-stage final-label rule
def build_two_stage_model(binary_pipeline, attack_pipeline) -> onnx.ModelProto:
    # Confirm both models were trained with same number of features
    binary_features = int(binary_pipeline.n_features_in_)
    attack_features = int(attack_pipeline.n_features_in_)
    if binary_features != attack_features:
        raise ValueError(
            "Binary and attack models use different feature counts: "
            f"{binary_features} and {attack_features}"
        )

    # Confirm both models had the same order of features
    binary_names = list(binary_pipeline.feature_names_in_)
    attack_names = list(attack_pipeline.feature_names_in_)
    if binary_names != attack_names:
        raise ValueError("Binary and attack models use different feature orderings")

    # Convert and namesapce the two fitted models
    binary = prefix_stage(
        convert_stage(binary_pipeline, "binary_input"), "binary_", "binary_input"
    )
    attack = prefix_stage(
        convert_stage(attack_pipeline, "attack_input"), "attack_", "attack_input"
    )

    # Construct graph containing the nodes and fitted parameters from both classifiers
    graph = helper.make_graph(
        nodes=list(binary.graph.node) + list(attack.graph.node),
        name="two_stage_intrusion_detection_system",
        inputs=[
            helper.make_tensor_value_info(
                "network_features", TensorProto.FLOAT, [None, binary_features]
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "final_prediction", TensorProto.INT64, [None]
            ),
            helper.make_tensor_value_info(
                "stage_1_prediction", TensorProto.INT64, [None]
            ),
            helper.make_tensor_value_info(
                "stage_1_probabilities", TensorProto.FLOAT, [None, 2]
            ),
            helper.make_tensor_value_info(
                "stage_2_prediction", TensorProto.INT64, [None]
            ),
            helper.make_tensor_value_info(
                "stage_2_probabilities", TensorProto.FLOAT, [None, 9]
            ),
        ],

        # Include the fitted scalar and forest parameters from both converted models
        initializer=list(binary.graph.initializer)
        + list(attack.graph.initializer)
        + [helper.make_tensor("normal_label", TensorProto.INT64, [1], [0])],
        value_info=list(binary.graph.value_info) + list(attack.graph.value_info),
    )

    # Determine whether Stage 1 predicted normal
    # Select 0 for normal rows
    # Expose the individual stage predictions and probabilities for diagnostics
    graph.node.extend(
        [
            # Produce boolean true wherever Staeg 1 predicted label 0
            helper.make_node(
                "Equal",
                ["binary_label", "normal_label"],
                ["is_normal"],
                name="route_normal_traffic",
            ),
            # Apply final routing decision
            helper.make_node(
                "Where",
                ["is_normal", "normal_label", "attack_label"],
                ["final_prediction"],
                name="select_final_label",
            ),
            helper.make_node(
                "Identity",
                ["binary_label"],
                ["stage_1_prediction"],
            ),
            helper.make_node(
                "Identity",
                ["binary_probabilities"],
                ["stage_1_probabilities"],
            ),
            helper.make_node(
                "Identity",
                ["attack_label"],
                ["stage_2_prediction"],
            ),
            helper.make_node(
                "Identity",
                ["attack_probabilities"],
                ["stage_2_probabilities"],
            ),
        ]
    )

    # Combine the operator-set requirements from both converted model graphs
    opsets = {}
    for model in (binary, attack):
        for opset in model.opset_import:
            opsets[opset.domain] = max(opsets.get(opset.domain, 0), opset.version)

    # Wrap the completed graph in an ONNX metadata for future validation and deployment tooling
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid(domain, version) for domain, version in opsets.items()],
        producer_name="IntrusionDetectionSystemProject",
    )
    model.ir_version = min(binary.ir_version, attack.ir_version)
    model.metadata_props.add(
        key="feature_names", value="|".join(str(name) for name in binary_names)
    )

    # Run Onnx's structural validation before writing model to disk
    onnx.checker.check_model(model)
    return model

# Create a Triton config for the ONNX Runtime backend
def write_triton_config(number_of_features: int) -> None:
    # Build the protobuf-text configuration using the fitted feature count
    config = f'''name: "two_stage_ids"
platform: "onnxruntime_onnx"
max_batch_size: 0

instance_group [
  {{
    count: 1
    kind: KIND_CPU
  }}
]

input [
  {{
    name: "network_features"
    data_type: TYPE_FP32
    dims: [-1, {number_of_features}]
  }}
]

output [
  {{ name: "final_prediction" data_type: TYPE_INT64 dims: [-1] }},
  {{ name: "stage_1_prediction" data_type: TYPE_INT64 dims: [-1] }},
  {{ name: "stage_1_probabilities" data_type: TYPE_FP32 dims: [-1, 2] }},
  {{ name: "stage_2_prediction" data_type: TYPE_INT64 dims: [-1] }},
  {{ name: "stage_2_probabilities" data_type: TYPE_FP32 dims: [-1, 9] }}
]
'''
    # Create Triton model directory if it doesn't exist already
    TRITON_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save model configuration as UTF-8 test
    TRITON_CONFIG_PATH.write_text(config, encoding="utf-8")

# Load validation rows using the same cleaning rules as normal inference
def clean_sample(dataset_path: Path, sample_size: int) -> pd.DataFrame:
    # Load the validation feature dataset
    features = pd.read_csv(dataset_path)

    # Removes the "Label" column in Dataset.csv is there is any.
    if "Label" in features.columns:
        features = features.drop(columns=["Label"])

    # Treats all positive/negative infinite values as a missing value
    features = features.replace([np.inf, -np.inf], np.nan)

    # Replace missing numeric values with the median of their feature column
    features = features.fillna(features.median(numeric_only=True))

    # Limit validation to the requested number of rows
    return features.head(sample_size)

# Produce reference labels using the original Python two-stage behavior
def expected_two_stage_labels(
    sample: pd.DataFrame, binary_pipeline, attack_pipeline
) -> Tuple[np.ndarray, np.ndarray]:
    # Run Stage 1 and ensure its labels use the same INT64 type as ONNX
    binary_labels = np.asarray(binary_pipeline.predict(sample), dtype=np.int64)

    # Initialize every final prediction as normal traffic
    expected = np.zeros_like(binary_labels)

    # Route according to Stage 1 predictions
    attack_mask = binary_labels == 1

    # Run Stage 2 if btach contains predicted attacks
    if attack_mask.any():
        expected[attack_mask] = attack_pipeline.predict(sample.loc[attack_mask])
    return binary_labels, expected

# Check ONNX structure and predictions against the Python implementation
def validate_export(binary_pipeline, attack_pipeline, sample_size: int = 1000) -> None:
    # Load the exported model through the same ONNX Runtime backend used by Triton
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    # Prepare a validation sample and convert it to FP32 input type expected by the ONNX graph
    sample_df = clean_sample(PROJECT_ROOT / "Dataset.csv", sample_size)
    sample = sample_df.to_numpy(dtype=np.float32)

    # Generate reference predictions with the original Python implementation
    expected_binary, expected_final = expected_two_stage_labels(
        sample_df, binary_pipeline, attack_pipeline
    )

    # Run all exported ONNX outputs and associate each tensor with its output name
    outputs = dict(
        zip(
            [output.name for output in session.get_outputs()],
            session.run(None, {"network_features": sample}),
        )
    )

    # Measure the fraction of labels that exactly match between implementations
    binary_agreement = np.mean(outputs["stage_1_prediction"] == expected_binary)
    final_agreement = np.mean(outputs["final_prediction"] == expected_final)
    print(f"Binary label agreement: {binary_agreement:.4%}")
    print(f"Final label agreement:  {final_agreement:.4%}")

    # Treat prediction disagreement as an export failure
    if binary_agreement < 0.999 or final_agreement < 0.999:
        raise RuntimeError("ONNX predictions do not match the Python two-stage model")

    # Print the interface the clients and Triton must use
    print("\nONNX interface:")
    for value in session.get_inputs():
        print(f"  input  {value.name}: {value.type} {value.shape}")
    for value in session.get_outputs():
        print(f"  output {value.name}: {value.type} {value.shape}")

# Load both models, export the combined ONNX graph, generate the Triton configuration, validate the completed
# deployment artifact
def main() -> None:
    # Load the fitted binary and attack-only models from the disk
    binary_pipeline = joblib.load(MODEL_DIR / "random_forest_binary_ids.pkl")
    attack_pipeline = joblib.load(MODEL_DIR / "random_forest_multiclass_ids.pkl")

    # Build the combined two-stage ONNX model in memory
    model = build_two_stage_model(binary_pipeline, attack_pipeline)

    # Create Triton's model directory and save model.onnx
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, ONNX_PATH)

    # Generate config.pbtxt using the trained feature count
    write_triton_config(int(binary_pipeline.n_features_in_))

    print(f"Saved ONNX model:   {ONNX_PATH}")
    print(f"Saved Triton config: {TRITON_CONFIG_PATH}")

    # Confirm the same artifact loads and matches the original predictions
    validate_export(binary_pipeline, attack_pipeline)


if __name__ == "__main__":
    main()
