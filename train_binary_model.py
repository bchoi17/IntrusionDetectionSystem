"""
XGBoost_model.py

This script trains a binary Intrusion Detecction System (IDS) model

The model trained is a binary Random Forest classification model within a Pipeline. It is used to predict whether a
traffic flow is normal or an attack based on the traffic features given in the datasets. However, if the network flow
is an attack, the model cannot specify the type of attack.

Datasets used:
- Dataset.csv contains the network-flow features
- Label.csv contains the numeric class label for each row
"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report, confusion_matrix,
)

# Loads data and checks if both datasets have the same number of rows
def load_data(dataset_path, label_path):
    X = pd.read_csv(dataset_path)
    y_raw = pd.read_csv(label_path)

    # Shows the number of rows for Dataset and Label
    print("Dataset shape:", X.shape)
    print("Label shape:", y_raw.shape)

    # Checks if both datasets have the same number of rows
    # Error message will send if not the same number of rows
    if len(X) != len(y_raw):
        raise ValueError("Dataset and Label do not have the same number of rows.")

    y_multiclass = y_raw["Label"]

    # Binary Target
    # 0 = Normal
    # 1-9 = Attack
    y_binary = (y_multiclass != 0).astype(int)

    # Shows the orginal distribution of all attacks and normal flow
    print("\n Original label distribution:")
    print(y_multiclass.value_counts().sort_index())

    # Shows the binary distribution
    print("\n Binary Label distribution:")
    print(y_binary.value_counts().sort_index())

    return X, y_binary, y_multiclass

# Cleans data before in use
def clean_features(X):
    X = X.copy()

    # Replace infinite values if in dataset
    if "Label" in X.columns:
        X = X.drop(columns=["Label"])

    # Fill missing values with column median
    X = X.fillna(X.median(numeric_only=True))

    return X

# Builds the pipeline to build the Random Forest Classifier
def build_model():
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("oversampling", SMOTE(random_state=42, n_jobs=-1)),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ))
        ]
    )
    return model

# Evaluates the trained model based on Accuracy, Balanced Accuracy, Precision, Recall and F1 Score
# Also gives the Confusion Matrix and the Classification Report
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Shows the Accuracy, Balanced Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and
    # Classification Report
    print("\n Accuracy:", accuracy_score(y_test, y_pred))
    print("\n Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
    print("\n Precision:", precision_score(y_test, y_pred))
    print("\n Recall:", recall_score(y_test, y_pred))
    print("\n F1 Score:", f1_score(y_test, y_pred))
    print("\n Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))


def main():
    # Path of Dataset and Label csv
    dataset_path = "Dataset.csv"
    label_path = "Label.csv"

    # Loads the dataset and cleans the data
    X, y_binary, y_multiclass = load_data(dataset_path, label_path)
    X = clean_features(X)

    # Train and test the data
    X_train, X_test, y_train, y_test = train_test_split(
X,
        y_binary,
        test_size=0.2,
        random_state=42,
        stratify=y_binary
    )

    # Shows the number of training and testing rows
    print("\nTraining rows:", X_train.shape[0])
    print("\nTesting rows:", X_test.shape[0])

    # Builds the model
    model = build_model()

    # Trains the model
    print("\nTraining binary Random Forest IDS model")
    model.fit(X_train, y_train)

    # Evaluates the model
    print("\nEvaluating model")
    evaluate_model(model, X_test, y_test)

    # Saves model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest_binary_ids.pkl")

    # Save the feature names separately for reature importance and SHAP later
    joblib.dump(list(X.columns), "models/feature_names.pkl")

    # Confirms that the script ran completely
    print("\nSaved model to models/random_forest_binary_ids.pkl")
    print("Saved feature names to models/feature_names.pkl")

if __name__ == "__main__":
    main()