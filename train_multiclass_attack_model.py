"""
train_multiclass_attack_model.py

This script trains a multiclass attack-type classifier.

The model trained is a multiclass Random Forest classification model within a Pipeline. This model is used to predict
what the attack network flow's label is based on the 1-9 attack categories. This script also removes all normal network
flows in the dataset since this model is only going to work with the attack network flows.

Label 0 is normal traffic and will be removed
Labels 1-9 are attack categories

Datasets used:
- Dataset.csv contains the network-flow features
- Label.csv contains the numeric class label for each row
"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from skl2onnx import __max_supported_opset__, to_onnx
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from train_binary_model import clean_features

"""
Function is used to load the two datasets the model need to train/test

- Checks if the two datasets has the same number of rows 
- Shows the distribution of that attack and normal label distribution
"""
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

    y = y_raw["Label"]

    # Shows the binary distribution of attack and normal network flow
    print("\nLabel distribution:")
    print(y.value_counts().sort_index())

    return X, y

""" 
# Cleans the data so we are left with only the attack types
def clean_features(X):
    X = X.copy()

    # If there is a "Label" column in Dataset, the column is removed
    if "Label" in X.columns:
        X = X.drop(columns=["Label"])

    # Fill missing values with column median
    X = X.fillna(X.median(numeric_only=True))

    return X
"""

"""
Filters out the normal network flow rows to only have the attack network flow rows in the datasets
"""
def filter_attack_rows(X, y):
    # Filters out normal network flows in both Dataset.csv and Label.csv
    attack_mask = y != 0
    X_attack = X.loc[attack_mask].copy()
    y_attack = y.loc[attack_mask].copy()

    # Prints the number of rows the attack-only dataset contains and the attack label distribution
    print("\nAttack-only dataset shape", X_attack.shape)
    print("Attack label distribution")
    print(y_attack.value_counts().sort_index())

    return X_attack, y_attack

"""
Builds a random forest model

- StandardScaler(): Standardize dataset to center mean as 0 and standard deviation as 1
- SMOTE(): Fixes imbalance in the dataset
- RandomForestClassifier: The Random Forest Classifier model
"""
def build_model():
    model = Pipeline(
        steps = [
            ("scaler", StandardScaler()),
            ("oversampler", RandomOverSampler(
                sampling_strategy="not majority",
                random_state=42
            )),
            ("classifier", RandomForestClassifier(
                n_estimators=500, # 500 decision trees used
                max_depth=30,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1
            ))
        ]
    )
    return model

"""
Evaluates the trained model based on 
- Accuracy
- Balanced Accuracy
- Confusion Matrix
"""
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Shows the Accuracy, Balanced Accuracy, Confusion Matrix, and Classification Report
    print("\n Accuracy:", accuracy_score(y_test, y_pred))
    print("\n Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
    #print("\n Precision:", precision_score(y_test, y_pred))
    #print("\n Recall:", recall_score(y_test, y_pred))
    #print("\n F1 Score:", f1_score(y_test, y_pred))
    print("\n Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

def main():
    # Path of Dataset and Label
    dataset_path = "Dataset.csv"
    label_path = "Label.csv"

    # Loads and cleans the dataset
    X, y = load_data(dataset_path, label_path)
    X = clean_features(X)
    X_attack, y_attack = filter_attack_rows(X, y)

    # Trains and tests the clean data
    X_train, X_test, y_train, y_test = train_test_split(
        X_attack,
        y_attack,
        test_size=0.2,
        random_state=42,
        stratify=y_attack
    )

    # Shows the number of rows in trained/tested dataset
    print("\nTraining rows:", X_train.shape[0])
    print("\nTesting rows:", X_test.shape[0])

    # Builds the model
    model = build_model()

    # Trains the model
    print("\nTraining multiclass Random Forest IDS model")
    model.fit(X_train, y_train)

    # Evaluates the model
    print("\nEvaluating model")
    evaluate_model(model, X_test, y_test)

    # Saves model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest_multiclass_ids.pkl")

    # Save the feature names separately for reature importance and SHAP later
    joblib.dump(list(X.columns), "models/multiclass_feature_names.pkl")

    # Confirms that the script ran completely
    print("\nSaved model to models/random_forest_multiclass_ids.pkl")
    print("Saved feature names to models/multiclass_feature_names.pkl")


if __name__ == "__main__":
    main()


    X = pd.read_csv(dataset_path)
    y_raw = pd.read_csv(label_path)

    # Shows the number of rows for Dataset and Label
    print("Dataset shape:", X.shape)
    print("Label shape:", y_raw.shape)

    # Checks if both datasets have the same number of rows
    # Error message will send if not the same number of rows
    if len(X) != len(y_raw):
        raise ValueError("Dataset and Label do not have the same number of rows.")

    y = y_raw["Label"]

    # Shows the binary distribution
    print("\nLabel distribution:")
    print(y.value_counts().sort_index())

    return X, y

""" 
# Cleans the data so we are left with only the attack types
def clean_features(X):
    X = X.copy()

    # If there is a "Label" column in Dataset, the column is removed
    if "Label" in X.columns:
        X = X.drop(columns=["Label"])

    # Fill missing values with column median
    X = X.fillna(X.median(numeric_only=True))

    return X
"""

# Filters out the normal traffic rows to only have the attack rows
def filter_attack_rows(X, y):
    attack_mask = y != 0
    X_attack = X.loc[attack_mask].copy()
    y_attack = y.loc[attack_mask].copy()

    # Prints the number of rows the attack-only dataset and the attack label distribution
    print("\nAttack-only dataset shape", X_attack.shape)
    print("Attack label distribution")
    print(y_attack.value_counts().sort_index())

    return X_attack, y_attack

# builds a random forest model
def build_model():
    """
    model = Pipeline(
        steps = [
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ]
    )
    """
    model = Pipeline(
        steps = [
            ("scaler", StandardScaler()),
            ("oversampler", RandomOverSampler(
                sampling_strategy="not majority",
                random_state=42
            )),
            ("classifier", RandomForestClassifier(
                n_estimators=500,
                max_depth=30,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1
            ))
        ]
    )
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Shows the Accuracy, Balanced Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and
    # Classification Report
    print("\n Accuracy:", accuracy_score(y_test, y_pred))
    print("\n Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
    #print("\n Precision:", precision_score(y_test, y_pred))
    #print("\n Recall:", recall_score(y_test, y_pred))
    #print("\n F1 Score:", f1_score(y_test, y_pred))
    print("\n Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

def main():
    # Path of Dataset and Label
    dataset_path = "Dataset.csv"
    label_path = "Label.csv"

    # Loads and cleans the data
    X, y = load_data(dataset_path, label_path)
    X = clean_features(X)
    X_attack, y_attack = filter_attack_rows(X, y)

    # Trains and tests the clean data
    X_train, X_test, y_train, y_test = train_test_split(
        X_attack,
        y_attack,
        test_size=0.2,
        random_state=42,
        stratify=y_attack
    )

    # Shows the number of rows in trained/tested dataset
    print("\nTraining rows:", X_train.shape[0])
    print("\nTesting rows:", X_test.shape[0])

    # Builds the model
    model = build_model()

    # Trains the model
    print("\nTraining multiclass Random Forest IDS model")
    model.fit(X_train, y_train)

    # Evaluates the model
    print("\nEvaluating model")
    evaluate_model(model, X_test, y_test)

    # Saves model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest_multiclass_ids.pkl")

    # Save the feature names separately for reature importance and SHAP later
    joblib.dump(list(X.columns), "models/multiclass_feature_names.pkl")

    # Confirms that the script ran completely
    print("\nSaved model to models/random_forest_multiclass_ids.pkl")
    print("Saved feature names to models/multiclass_feature_names.pkl")


if __name__ == "__main__":
    main()


