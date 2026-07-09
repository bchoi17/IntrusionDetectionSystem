"""
train_multiclass_full_model.py

This script builds a model using all 10 labels in the Label.csv to predict what traffic flow it is instead of only
using the attack labels.

The model itself is a Random Forest Multiclassifier than can be evaluated based on the accuracy, balanced, accuracy,
the confusion matrix, and the classification report.
"""


import os
import joblib
import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

# Loads both datasets, checks if both csvs have same number of rows, and shows the label distribution
def load_data(dataset_path, label_path):
    # Load Dataset and Label csv
    X = pd.read_csv(dataset_path)
    y_raw = pd.read_csv(label_path)

    # Shows the number of rows in Dataset and Label
    print("Dataset shape: ", X.shape)
    print("Label shape: ", y_raw.shape)

    # Checks whether Dataset and Label have same number of rows
    if len(X) != len(y_raw):
        raise ValueError("Dataset and Label do not have the same number of rows.")

    y = y_raw["Label"]

    # Shows the distribution of the labels
    print("\nOriginal Label Distribution:")
    print(y.value_counts().sort_index())

    return X, y

# Cleans the feature dataset
def clean_features(X):
    X = X.copy()

    # Removes "Label" column in Dataset if there is one
    if "Label" in X.columns:
        X = X.drop(columns=["Label"])

    # Replaces infinite values with np.nan
    X = X.replace([np.inf, -np.inf], np.nan)

    # Replaces missing vlaues with median of column
    X.fillna(X.median(numeric_only=True))

    return X

# Builds a multiclass Random Forest Model with oversampling to combat imbalanced data
def build_model():
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
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

# Shows the accuracy, balanced accuracy, confusion matrix, and classification report
def evaluate_model(model, X_test, y_test):
    # predict the model
    y_pred = model.predict(X_test)

    # Shows the accuracy and balanced accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    print("\nAccuracy:", accuracy)
    print("\nBalanced Accuracy:", balanced_accuracy)

    # Shows thw confusion matrix and the classification report
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    print(classification_report(y_test, y_pred, zero_division=0))

    return accuracy, balanced_accuracy, report

# Saves metrics/results into a csv file
def save_results(accuracy, balanced_accuracy, report):
    os.makedirs("models", exist_ok=True)
    report_df = pd.DataFrame(report).transpose()
    report_df["overall_accuracy"] = accuracy
    report_df["overall_balanced_accuracy"] = balanced_accuracy

    report_df.to_csv("models/multiclass_oversampled_results.csv")

    print("\nSaved results to models/multiclass_oversampled_results.csv")


def main():
    print("Starting oversampled multiclass Random Forest training...")

    dataset_path = "Dataset.csv"
    label_path = "Label.csv"

    X, y = load_data(dataset_path, label_path)
    X = clean_features(X)

    print("\nSplitting data into train and test sets...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training rows:", X_train.shape[0])
    print("Testing rows:", X_test.shape[0])

    print("\nTraining label distribution before oversampling:")
    print(y_train.value_counts().sort_index())

    model = build_model()

    print("\nTraining oversampled multiclass Random Forest...")
    model.fit(X_train, y_train)

    accuracy, balanced_accuracy, report = evaluate_model(model, X_test, y_test)

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/random_forest_multiclass_oversampled.pkl")
    joblib.dump(list(X.columns), "models/multiclass_oversampled_feature_names.pkl")

    save_results(accuracy, balanced_accuracy, report)

    print("\nSaved model to models/random_forest_multiclass_oversampled.pkl")
    print("Saved feature names to models/multiclass_oversampled_feature_names.pkl")
    print("\nScript completed successfully.")


if __name__ == "__main__":
    main()







