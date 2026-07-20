"""
two_stage_ids_predict.py

Ths script will use two models in train_binary_model.py and train_multiclass_attack_model.py in order to predict the
type of label based on the network features. This will be run in two stages,

The first model (train_binary_model.py) will be a binary Random Forest classification model that will only predict if
the network flow is a normal or attack label. However, the model will not specify the attack label.

The second model (train_multiclass_attack_model.py) will be a multiclass Random Forest classification model that will
only predict what the attack label is if the binary Random Forest classification model predicts the network flow to be
an attack label.

Stage 1: train_binary_model.py
    - This will be the first model in use to predict if the network flow is normal or an attack
    - If the network flow is normal, then the network flow does not move on to Stage 2
    - If the network flow is an attack, then the network flow moves on to Stage 2

Stage 2: train_multiclass_model.py
    - This will predict what type of attack is the network flow basd on the 9 attack labels in the dataset

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

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

"""
Function is used to load the two datasets the model need to train/test

- Checks if the two datasets has the same number of rows 
- Shows the distribution of that attack and normal label distribution
"""
def clean_features(X):
    X = X.copy()

    # If Dataset.csv has a "Label" column, then remove the column
    if "Label" in X.columns:
        X = X.drop(columns=["Label"])

    # Treats all positive/negative infinite values as a missing value
    X = X.replace([np.inf, -np.inf], np.nan)

    # Replace missing numeric values with the median of their feature column
    X = X.fillna(X.median(numeric_only=True))

    return X

"""
Loads the binary and attack-only multiclass models
"""
def load_models():
    # Loads the two models
    binary_model = joblib.load("models/random_forest_binary_ids.pkl")
    attack_model = joblib.load("models/random_forest_multiclass_ids.pkl")

    return binary_model, attack_model

"""
Runs both of the models in two stages to make predictions
"""
def run_two_stage_predictions(X, binary_model, attack_model):
    # Signal Stage 1 of predictions is starting
    print("Running Stage 1 binary predictions...")

    # Binary model predicts if the row is a normal or an attack network flow
    # Binary model gives the probability of the prediction
    binary_predictions = binary_model.predict(X)
    binary_probabilities = binary_model.predict_proba(X)

    # Both the normal and attack flow probability is stored in these variables
    normal_probabilities = binary_probabilities[:, 0]
    attack_probabilities = binary_probabilities[:, 1]

    # Data frame of the binary model predictions, normal probabilities, and attack probabilities
    results_df = pd.DataFrame({
        "row_index": X.index,
        "stage_1_prediction": binary_predictions,
        "normal_probability": normal_probabilities,
        "attack_probability": attack_probabilities
    })

    # Lists the normal and attack network flows as 0 and 1
    results_df["stage_1_label"] = results_df["stage_1_prediction"].map({
        0: "Normal",
        1: "Attack"
    })

    # Default values for rows predicted as a normal network flow
    results_df["stage_2_attack_label"] = None
    results_df["stage_2_attack_confidence"] = None
    results_df["final_prediction"] = "Normal Traffic"

    # Only sends predicted attack network flows into the attack classifier
    attack_mask = binary_predictions == 1
    X_predicted_attacks = X.loc[attack_mask]

    # Prints the number of rows that were predicted for normal and attack
    print("Rows predicted as attack:", X_predicted_attacks.shape[0])
    print("Rows predicted as normal:", X.shape[0] - X_predicted_attacks.shape[0])

    # If the network flow was considered an attack, the row will move on to Stage 2
    if len(X_predicted_attacks) > 0:
        # Signal Stage 2 of predictions is starting
        print("Running Stage 2 attack-type predictions...")

        # Predicts the type of attack the attack network flow is and the probability of the predicted
        # attack label being true
        attack_predictions = attack_model.predict(X_predicted_attacks)
        attack_probabilities_stage_2 = attack_model.predict_proba(X_predicted_attacks)
        attack_confidences = attack_probabilities_stage_2.max(axis=1)

        # Dataframe of attack predictions and probabilities
        results_df.loc[attack_mask, "stage_2_attack_label"] = attack_predictions
        results_df.loc[attack_mask, "stage_2_attack_confidence"] = attack_confidences
        results_df.loc[attack_mask, "final_prediction"] = [
            f"Attack Label {label}" for label in attack_predictions
        ]

    return results_df

"""
Adds a True/False label to show if the prediction of the data was correct
"""
def add_true_labels(results_df, y):
    results_df["true_label"] = y.values

    results_df["true_binary_label"] = (results_df["true_label"] != 0).astype(int)

    results_df["binary_correct"] = (
        results_df["stage_1_prediction"] == results_df["true_binary_label"]
    )

    results_df["final_correct"] = False

    # If true label is 0, final prediction should be Normal Traffic.
    normal_correct = (
        (results_df["true_label"] == 0) &
        (results_df["final_prediction"] == "Normal Traffic")
    )

    # If true label is nonzero, final attack label should match true label.
    attack_correct = (
        (results_df["true_label"] != 0) &
        (results_df["stage_2_attack_label"] == results_df["true_label"])
    )

    results_df["final_correct"] = normal_correct | attack_correct

    return results_df

"""
Print summary information of the two-stage prediction model
"""
def print_summary(results_df):
    print("\nTwo-Stage IDS Summary")
    print("---------------------")

    # Prints the predicted counts of normal/attack label
    print("\nPredicted Stage 1 counts:")
    print(results_df["stage_1_label"].value_counts())

    # Grabs and prints the true counts of attack/normal label
    if "true_label" in results_df.columns:
        true_binary_counts = results_df["true_binary_label"].map({
            0: "Normal",
            1: "Attack"
        }).value_counts()

        print("\nTrue Stage 1 counts:")
        print(true_binary_counts)

    # Prints the predicted counts each attack label
    #print("\nPredicted final class counts:")
    #print(results_df["final_prediction"].value_counts())

    # Grabs and prints the true counts of each attack label
    # Also compares the difference between each predicted and true attack label count
    # Shows first stage accuracy and full stage accuracy
    # Shows the count of missed attacks and false alarms
    # Also gives other forms of measures of first stage and full stage such as precision, recall, f1-score, and balanced accuracy
    if "true_label" in results_df.columns:
        true_final_labels = results_df["true_label"].apply(
            lambda label: "Normal Traffic" if label == 0 else f"Attack Label {label}"
        )

        #print("\nTrue final class counts:")
        #print(true_final_labels.value_counts())

        # Stores the number of predicted and true attack network flows
        predicted_counts = results_df["final_prediction"].value_counts()
        true_counts = true_final_labels.value_counts()

        # Data frame of the predicted and true attack network flows
        comparison_df = pd.DataFrame({
            "true_count": true_counts,
            "predicted_count": predicted_counts
        }).fillna(0).astype(int)

        # Compares the number of predicted and true attack network flows
        comparison_df["difference"] = (comparison_df["predicted_count"] - comparison_df["true_count"])
        print("\nTrue vs Predicted final class counts:")
        print(comparison_df)

        # Show the accuracy of Stage 1 and Stage 2 predictions
        binary_accuracy = results_df["binary_correct"].mean()
        final_accuracy = results_df["final_correct"].mean()
        print("\nBinary stage accuracy:", binary_accuracy)
        print("Full two-stage final accuracy:", final_accuracy)

        # Shows how many missed attack network flows there are
        missed_attacks = results_df[
            (results_df["true_label"] != 0) &
            (results_df["stage_1_prediction"] == 0)
            ]

        # Shows how many normal network flows were predicted as an attack
        false_alarms = results_df[
            (results_df["true_label"] == 0) &
            (results_df["stage_1_prediction"] == 1)
            ]
        print("Missed attacks at Stage 1:", len(missed_attacks))
        print("False alarms at Stage 1:", len(false_alarms))

        # Shows the Stage 1 accuracy, balanced accuracy, precision, recall, and F1 Score
        y_true_binary = results_df["true_binary_label"]
        y_pred_binary = results_df["stage_1_prediction"]
        print("\nBinary Stage Metrics:")
        print("Accuracy:", accuracy_score(y_true_binary, y_pred_binary))
        print("Balanced Accuracy:", balanced_accuracy_score(y_true_binary, y_pred_binary))
        print("Precision:", precision_score(y_true_binary, y_pred_binary))
        print("Recall:", recall_score(y_true_binary, y_pred_binary))
        print("F1 Score:", f1_score(y_true_binary, y_pred_binary))

        # Shows the Stage 1 classification report
        print("\nBinary Stage Classification Report:")
        print(classification_report(
            y_true_binary,
            y_pred_binary,
            target_names=["Normal", "Attack"],
            zero_division=0
        ))

        # Final Two-Stage Multiclass Metrics
        y_true_final = results_df["true_label"]

        """
        This function converts the final predictions into labels 
        """
        def convert_final_prediction_to_label(prediction):
            # If the prediction is a normal network flow, then converts to 0
            if prediction == "Normal Traffic":
                return 0

            # If the prediction is an attack network flow, then converts to "Attack Label"
            return int(prediction.replace("Attack Label ", ""))

        # Converts the predictions into labels using the convert_to_final_prediction_to_label
        y_pred_final = results_df["final_prediction"].apply(
            convert_final_prediction_to_label
        )

        # Shows the two-stage model accuracy, balanced accuracy, macro precision, macro recall, macro F1 score, and
        # weighted F1 score
        print("\nFinal Two-Stage Multiclass Metrics:")
        print("Accuracy:", accuracy_score(y_true_final, y_pred_final))
        print("Balanced Accuracy:", balanced_accuracy_score(y_true_final, y_pred_final))
        print("Macro Precision:", precision_score(y_true_final, y_pred_final, average="macro", zero_division=0))
        print("Macro Recall:", recall_score(y_true_final, y_pred_final, average="macro", zero_division=0))
        print("Macro F1 Score:", f1_score(y_true_final, y_pred_final, average="macro", zero_division=0))
        print("Weighted F1 Score:", f1_score(y_true_final, y_pred_final, average="weighted", zero_division=0))

        # Shows the two-stage model classification report
        print("\nFinal Two-Stage Classification Report:")
        print(classification_report(
            y_true_final,
            y_pred_final,
            zero_division=0
        ))

        # Shows the two-stage model confusion matrix
        print("\nFinal Two-Stage Confusion Matrix:")
        print(confusion_matrix(y_true_final, y_pred_final))

def main():
    print("Starting two-stage IDS prediction...")

    # Path for Dataset.csv and Label.csv
    dataset_path = "Dataset.csv"
    label_path = "Label.csv"

    # Load Dataset.csv and Label.csv
    X = pd.read_csv(dataset_path)
    y = pd.read_csv(label_path)["Label"]

    # Cleans Dataset
    X = clean_features(X)

    binary_model, attack_model = load_models()

    # You can run all rows, but for quick testing, start with a sample.
    # Change this to None if you want to predict the full dataset.
    sample_size = None

    if sample_size is not None:
        X_batch = X.sample(n=sample_size, random_state=42)
        y_batch = y.loc[X_batch.index]
    else:
        X_batch = X
        y_batch = y

    print("Dataset size:", X_batch.shape[0])

    results_df = run_two_stage_predictions(
        X_batch,
        binary_model,
        attack_model
    )

    results_df = add_true_labels(results_df, y_batch)

    print_summary(results_df)

    os.makedirs("models/models", exist_ok=True)

    output_path = "models/two_stage_ids_predictions.csv"
    results_df.to_csv(output_path, index=False)

    print("\nSaved predictions to:", output_path)
    print("Script completed successfully.")


if __name__ == "__main__":
    main()