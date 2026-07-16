"""
client.py

This script tests the deployed two-stage IDS model through the Triton Inference Server.

This script also reads the network flow records from Dataset.csv, cleans and converts the 76 features in Dataset.csv to
FP32, sends them to the two_stage_ids model, and prints the binary stage probabilities and final predictions.

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


# Store the directory containing this script so Dataset.csv can be located regardless of the directory from which the
# client is launched.
PROJECT_ROOT = Path(__file__).resolve().parent

# These names must exactly match the model and tensor names in Triton's config.pbtxt and the exported ONNX model.
MODEL_NAME = "two_stage_ids"
INPUT_NAME = "network_features"
OUTPUT_NAMES = (
    "final_prediction",
    "stage_1_prediction",
    "stage_1_probabilities",
    "stage_2_prediction",
    "stage_2_probabilities",
)

def load_batch(dataset_path: Path, start_row: int, batch_size: int) -> np.ndarray:
    """Load and clean a batch using the same rules as local prediction.

    Parameters
    ----------
    dataset_path:
        Location of the CSV file containing network-flow features.
    start_row:
        Zero-based row number at which the requested batch starts.
    batch_size:
        Maximum number of network-flow rows to include in the batch.

    Returns
    -------
    numpy.ndarray
        A contiguous FP32 array shaped ``[number_of_rows, 76]`` that can be
        sent directly to the Triton model.
    """

    # Load all network-flow records and column names from the CSV file.
    features = pd.read_csv(dataset_path)

    # If Dataset.csv has a "Label" column, then remove the column
    if "Label" in features.columns:
        features = features.drop(columns=["Label"])

    # Treats all positive/negative infinite values as a missing value
    features = features.replace([np.inf, -np.inf], np.nan)

    # Replace missing numeric values with the median of their feature column
    features = features.fillna(features.median(numeric_only=True))

    # Select only the rows requested through --start-row and --batch-size.
    batch = features.iloc[start_row : start_row + batch_size]

    # Give a clear error instead of sending an empty request to Triton.
    if batch.empty:
        raise ValueError(f"No rows found at start row {start_row}")

    # Protect the model from a dataset with a missing or additional feature.
    if batch.shape[1] != 76:
        raise ValueError(f"Expected 76 features, found {batch.shape[1]}")

    # Convert pandas values to the FP32 tensor type expected by ONNX/Triton.
    # A contiguous array also ensures the values have a standard memory layout.
    return np.ascontiguousarray(batch.to_numpy(dtype=np.float32))


def request_json(url: str, payload=None):
    """Send one JSON request to Triton and decode its JSON response.

    A request without a payload uses HTTP GET, which is used for readiness
    checks. A request with a payload uses HTTP POST, which is used for model
    inference.
    """

    # Convert a Python dictionary into UTF-8 JSON bytes for an HTTP request.
    # Readiness GET requests do not have a request body, so data remains None.
    data = None if payload is None else json.dumps(payload).encode("utf-8")

    # Build either a GET or POST request. The Content-Type header is necessary
    # when an inference JSON body is included.
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        # Send the request and wait up to 60 seconds for Triton to respond.
        with urlopen(request, timeout=60) as response:
            # Read the complete HTTP response body as bytes.
            body = response.read()

            # Decode a non-empty JSON response. Triton readiness endpoints
            # commonly return an empty body when the readiness check succeeds.
            return json.loads(body) if body else None
    except (HTTPError, URLError) as error:
        # Convert lower-level connection and HTTP errors into a message that
        # identifies the exact Triton endpoint that failed.
        raise RuntimeError(f"Triton request failed for {url}: {error}") from error


def infer(server_url: str, batch: np.ndarray):
    """Send a feature batch to Triton and return every model output by name."""

    # Construct the base HTTP address, such as http://127.0.0.1:8000.
    base_url = f"http://{server_url}"

    # Confirm both the server and this specific model are ready before sending
    # what may be a larger inference request.
    request_json(f"{base_url}/v2/health/ready")
    request_json(f"{base_url}/v2/models/{MODEL_NAME}/ready")

    # Build a request using Triton's V2 inference protocol. The two-dimensional
    # batch is flattened for JSON transport while its original shape is sent
    # separately so Triton can reconstruct the input tensor.
    payload = {
        "inputs": [
            {
                "name": INPUT_NAME,
                "shape": list(batch.shape),
                "datatype": "FP32",
                "data": batch.reshape(-1).tolist(),
            }
        ],
        "outputs": [{"name": name} for name in OUTPUT_NAMES],
    }

    # Submit the batch to version 1 of the loaded model (Triton automatically
    # chooses the available version because none is explicitly specified).
    response = request_json(f"{base_url}/v2/models/{MODEL_NAME}/infer", payload)

    # Convert each returned JSON list back into a NumPy array with the shape
    # reported by Triton, then store it under its ONNX output name.
    results = {}
    for output in response["outputs"]:
        results[output["name"]] = np.asarray(output["data"]).reshape(output["shape"])

    return results


def main() -> None:
    """Parse command-line options, run inference, and print readable results."""

    # Define the command-line interface. All options have defaults, so running
    # ``python client.py`` is sufficient for a five-row local test.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="127.0.0.1:8000")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "Dataset.csv")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=5)

    # Read the values supplied by the user into the args object.
    args = parser.parse_args()

    # Prepare the selected CSV rows and send them to the Triton server.
    batch = load_batch(args.dataset, args.start_row, args.batch_size)
    predictions = infer(args.url, batch)

    # Convert Triton's tensor outputs into one readable dictionary per input
    # row. The row offset is added to start_row to recover its CSV row number.
    rows = []
    for offset in range(len(batch)):
        # Extract scalar predictions from the current position in the batch.
        stage_1 = int(predictions["stage_1_prediction"][offset])
        final = int(predictions["final_prediction"][offset])

        # Binary probabilities are ordered as [normal, attack].
        binary_probabilities = predictions["stage_1_probabilities"][offset]

        # Store the values used in the final table. Stage 1 is translated to a
        # readable word while final_prediction remains the numeric class 0-9.
        rows.append(
            {
                "dataset_row": args.start_row + offset,
                "stage_1": "Attack" if stage_1 == 1 else "Normal",
                "normal_probability": float(binary_probabilities[0]),
                "attack_probability": float(binary_probabilities[1]),
                "final_prediction": final,
            }
        )

    # Create a compact table and print probabilities with six decimal places.
    result = pd.DataFrame(rows)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nFinal label 0 means normal; labels 1-9 identify the attack type.")


# Run main only when this file is executed directly. Importing client.py from
# another Python module will make its functions available without sending data.
if __name__ == "__main__":
    main()
