"""
Morpheus pipeline for the Triton Inference Server two-stage IDS (CPU only)

This script creates a Morpheus pipeline that reads network flow records from Dataset.csv, validates and cleans the 76
model features, and sends the network flow records to the two-stage IDS model hosted by the Triton Inference Server.
The returned predictions and probabilities are added to the original dataset, but saved under a new CSV file named
morpheus_predictions.csv.

Pipeline: CSV Source -> feature cleanup -> Triton HTTP -> prediction CSV output.

This intentionally uses a CPU-compatible custom HTTP stage instead of the
GPU-oriented Morpheus inference/preprocessing stages.

Labels:
0 = Normal
1-9 = Attack Labels

1 = Analysis
2 = Backdoor
3 = DoS
4 = Exploits
5 = Fuzzers
6 = Generic
7 = Reconnaissance
8 = Shellcode
9 = Worms

Datasets used:
- Dataset.csv contains the network-flow features
- Label.csv contains the numeric class label for each row
"""

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from morpheus.config import Config, ExecutionMode
from morpheus.messages import MessageMeta
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.pipeline.stage_decorator import source, stage

# Resolve every file location relative to this script so the exporter works regardless of the terminal's working
# directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Name for the model loaded by Triton
MODEL_NAME = "two_stage_ids"

# ONNX/Triton input name that matched with the confgi.pbtxt
INPUT_NAME = "network_features"

# Model outputs requested from Triton
PREDICTION_OUTPUTS = (
    "final_prediction",
    "stage_1_prediction",
    "stage_1_probabilities",
    "stage_2_prediction",
)

# 
def request_json(url: str, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urlopen(request, timeout=120) as response:
            body = response.read()
            return json.loads(body) if body else None
    except (HTTPError, URLError) as error:
        raise RuntimeError(f"Triton request failed for {url}: {error}") from error

# Load the trained feature order and fixed replacement medians.
def load_feature_contract(reference_dataset: Path):
    reference = pd.read_csv(reference_dataset)
    if "Label" in reference.columns:
        reference = reference.drop(columns=["Label"])
    reference = reference.replace([np.inf, -np.inf], np.nan)
    if reference.shape[1] != 76:
        raise ValueError(f"Expected 76 reference features, found {reference.shape[1]}")
    return list(reference.columns), reference.median(numeric_only=True)


@source(execution_modes=(ExecutionMode.CPU,))
def csv_flow_source(filename: str, batch_size: int) -> MessageMeta:
    """Stream CSV network-flow records as Morpheus messages."""

    for dataframe in pd.read_csv(filename, chunksize=batch_size):
        yield MessageMeta(dataframe)


@stage(execution_modes=(ExecutionMode.CPU,))
def clean_flow_features(
    message: MessageMeta,
    *,
    feature_names: list,
    feature_medians: pd.Series,
) -> MessageMeta:
    """Enforce the model's 76-column contract and sanitize numeric values."""

    with message.mutable_dataframe() as dataframe:
        missing = [name for name in feature_names if name not in dataframe.columns]
        if missing:
            raise ValueError(f"Input is missing model features: {missing}")
        dataframe.loc[:, feature_names] = (
            dataframe.loc[:, feature_names]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(feature_medians)
            .astype(np.float32)
        )
    return message


@stage(execution_modes=(ExecutionMode.CPU,))
def triton_ids_inference(
    message: MessageMeta,
    *,
    server_url: str,
    feature_names: list,
) -> MessageMeta:
    """Call Triton's V2 HTTP API and append predictions to the message."""

    base_url = f"http://{server_url}"
    with message.mutable_dataframe() as dataframe:
        batch = np.ascontiguousarray(
            dataframe.loc[:, feature_names].to_numpy(dtype=np.float32)
        )
        payload = {
            "inputs": [
                {
                    "name": INPUT_NAME,
                    "shape": list(batch.shape),
                    "datatype": "FP32",
                    "data": batch.reshape(-1).tolist(),
                }
            ],
            "outputs": [{"name": name} for name in PREDICTION_OUTPUTS],
        }
        response = request_json(
            f"{base_url}/v2/models/{MODEL_NAME}/infer", payload
        )
        outputs = {output["name"]: output for output in response["outputs"]}

        final = np.asarray(outputs["final_prediction"]["data"], dtype=np.int64)
        stage_1 = np.asarray(
            outputs["stage_1_prediction"]["data"], dtype=np.int64
        )
        stage_1_probabilities = np.asarray(
            outputs["stage_1_probabilities"]["data"], dtype=np.float32
        ).reshape(-1, 2)
        stage_2 = np.asarray(
            outputs["stage_2_prediction"]["data"], dtype=np.int64
        )

        dataframe["ids_final_prediction"] = final
        dataframe["ids_stage_1_prediction"] = stage_1
        dataframe["ids_normal_probability"] = stage_1_probabilities[:, 0]
        dataframe["ids_attack_probability"] = stage_1_probabilities[:, 1]
        # Stage 2 is meaningful only when Stage 1 predicts attack.
        dataframe["ids_stage_2_prediction"] = np.where(stage_1 == 1, stage_2, 0)
    return message


@stage(execution_modes=(ExecutionMode.CPU,))
def csv_prediction_sink(
    message: MessageMeta,
    *,
    output_filename: str,
) -> MessageMeta:
    """Append each completed Morpheus batch to the output CSV."""

    output = Path(output_filename)
    dataframe = message.df
    dataframe.to_csv(output, mode="a", header=not output.exists(), index=False)
    return message


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "Dataset.csv")
    parser.add_argument(
        "--reference-dataset", type=Path, default=PROJECT_ROOT / "Dataset.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "morpheus_predictions.csv"
    )
    parser.add_argument("--triton-url", default="triton-ids:8000")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {args.output}; pass --overwrite to replace it"
            )
        args.output.unlink()

    request_json(f"http://{args.triton_url}/v2/health/ready")
    request_json(f"http://{args.triton_url}/v2/models/{MODEL_NAME}/ready")
    feature_names, feature_medians = load_feature_contract(args.reference_dataset)

    config = Config()
    config.execution_mode = ExecutionMode.CPU
    config.pipeline_batch_size = args.batch_size
    config.model_max_batch_size = args.batch_size

    pipeline = LinearPipeline(config)
    pipeline.set_source(
        csv_flow_source(config, filename=str(args.input), batch_size=args.batch_size)
    )
    pipeline.add_stage(
        clean_flow_features(
            config,
            feature_names=feature_names,
            feature_medians=feature_medians,
        )
    )
    pipeline.add_stage(
        triton_ids_inference(
            config,
            server_url=args.triton_url,
            feature_names=feature_names,
        )
    )
    pipeline.add_stage(
        csv_prediction_sink(config, output_filename=str(args.output))
    )
    pipeline.run()
    print(f"Morpheus predictions saved to {args.output}")


if __name__ == "__main__":
    main()
