"""Classical Threat Layer: NSL-KDD Threat Classifier.

Trains and evaluates classical machine learning models (Logistic Regression
and Random Forest) to classify network traffic as normal (0) or attack (1),
and produces threat probability scores for adaptive quantum security decisions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Standard NSL-KDD 41 feature column names + label + difficulty score
NSL_KDD_COLUMNS: List[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]

CATEGORICAL_FEATURES: List[str] = ["protocol_type", "service", "flag"]
NUMERICAL_FEATURES: List[str] = [
    c for c in NSL_KDD_COLUMNS if c not in CATEGORICAL_FEATURES + ["label", "difficulty"]
]

DATA_URLS = {
    "KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
    "KDDTest+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt",
}


def download_dataset_if_needed(data_dir: Union[str, Path] = "data") -> None:
    """Download NSL-KDD train and test files if not already present."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    for filename, url in DATA_URLS.items():
        dest = data_path / filename
        if not dest.exists() or dest.stat().st_size == 0:
            print(f"Downloading {filename} to {dest}...")
            try:
                urllib.request.urlretrieve(url, dest)
                print(f"Successfully downloaded {filename} ({dest.stat().st_size / (1024*1024):.2f} MB)")
            except Exception as e:
                if dest.exists():
                    dest.unlink()
                raise FileNotFoundError(
                    f"Failed to download {filename} from {url}. Error: {e}\n"
                    f"Please manually place '{filename}' in the '{data_dir}/' folder."
                ) from e


def load_nsl_kdd(
    data_dir: Union[str, Path] = "data",
    download_if_missing: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load NSL-KDD train and test sets, applying column names and binary labels.

    Args:
        data_dir: Directory containing KDDTrain+.txt and KDDTest+.txt.
        download_if_missing: If True, attempts to download datasets if not found.

    Returns:
        Tuple of (train_df, test_df) with binary 'target' column (0=normal, 1=attack).
    """
    data_path = Path(data_dir)
    train_file = data_path / "KDDTrain+.txt"
    test_file = data_path / "KDDTest+.txt"

    if not (train_file.exists() and test_file.exists()):
        if download_if_missing:
            download_dataset_if_needed(data_dir)
        else:
            raise FileNotFoundError(
                f"NSL-KDD dataset files not found in '{data_dir}'.\n"
                f"Expected: '{train_file}' and '{test_file}'.\n"
                f"Download from: https://github.com/defcom17/NSL_KDD or run with download_if_missing=True."
            )

    train_df = pd.read_csv(train_file, names=NSL_KDD_COLUMNS, header=None)
    test_df = pd.read_csv(test_file, names=NSL_KDD_COLUMNS, header=None)

    # Binary classification: 'normal' -> 0, all attack types -> 1
    train_df["target"] = (train_df["label"] != "normal").astype(int)
    test_df["target"] = (test_df["label"] != "normal").astype(int)

    return train_df, test_df


def build_preprocessor() -> ColumnTransformer:
    """Build Scikit-Learn ColumnTransformer for categorical and numerical features."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def train_and_evaluate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models_dir: Union[str, Path] = "models",
    seed: int = 42,
) -> Dict[str, Any]:
    """Train Logistic Regression and Random Forest models, evaluate, and save artifacts."""
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    preprocessor = build_preprocessor()

    # Model 1: Logistic Regression baseline
    lr_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs"),
            ),
        ]
    )

    # Model 2: Random Forest Classifier
    rf_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=16,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    results: Dict[str, Any] = {}

    models = {
        "Logistic Regression (Baseline)": lr_pipeline,
        "Random Forest Classifier": rf_pipeline,
    }

    print("\n" + "=" * 70)
    print("PHASE 2: TRAINING AND EVALUATING THREAT CLASSIFIERS")
    print("=" * 70)

    comparison_data = []

    for name, pipe in models.items():
        print(f"Training {name}...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "pipeline": pipe,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
            "y_proba": y_proba,
        }

        comparison_data.append(
            {
                "Model": name,
                "Accuracy": f"{acc:.4f}",
                "Precision": f"{prec:.4f}",
                "Recall": f"{rec:.4f}",
                "F1-Score": f"{f1:.4f}",
                "TN": cm[0, 0],
                "FP": cm[0, 1],
                "FN": cm[1, 0],
                "TP": cm[1, 1],
            }
        )

    # Print comparison table
    comp_df = pd.DataFrame(comparison_data)
    print("\n--- MODEL PERFORMANCE COMPARISON ON TEST SET ---")
    print(comp_df.to_string(index=False))
    print("-" * 70)

    for name in models:
        cm = results[name]["confusion_matrix"]
        print(f"\nConfusion Matrix [{name}]:")
        print(f"  [TN: {cm[0, 0]:5d} | FP: {cm[0, 1]:5d}]")
        print(f"  [FN: {cm[1, 0]:5d} | TP: {cm[1, 1]:5d}]")

    # Save the Random Forest pipeline and the fitted preprocessor
    rf_pipe_path = models_path / "rf_pipeline.joblib"
    preprocessor_path = models_path / "preprocessor.joblib"

    print(f"\nSaving Random Forest pipeline to: {rf_pipe_path}")
    joblib.dump(rf_pipeline, rf_pipe_path)

    # Extract and save fitted preprocessor from rf pipeline
    joblib.dump(rf_pipeline.named_steps["preprocessor"], preprocessor_path)
    print(f"Saving preprocessor to: {preprocessor_path}")

    return results


def get_attack_probability(
    sample_df: pd.DataFrame,
    model_path: Union[str, Path] = "models/rf_pipeline.joblib",
) -> Union[float, np.ndarray]:
    """Calculate threat/attack probability P(Attack) in [0, 1] for network samples.

    Args:
        sample_df: DataFrame containing the required NSL-KDD feature columns.
        model_path: Path to the trained pipeline joblib artifact.

    Returns:
        Probability of attack as a float (single sample) or numpy array (batch).

    Raises:
        FileNotFoundError: If the model artifact does not exist.
        ValueError: If sample_df is missing required feature columns.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            f"Please run 'python src/classifier.py' first to train and save the model."
        )

    pipeline: Pipeline = joblib.load(path)
    required_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

    missing = [col for col in required_cols if col not in sample_df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required feature columns: {missing[:5]}... "
            f"(Total missing: {len(missing)})"
        )

    X = sample_df[required_cols]
    probabilities = pipeline.predict_proba(X)[:, 1]

    if len(probabilities) == 1:
        return float(probabilities[0])
    return probabilities


def load_sample_row(
    index: Optional[int] = None,
    split: str = "test",
    label_type: Optional[str] = None,
    data_dir: Union[str, Path] = "data",
) -> Tuple[pd.DataFrame, int, str]:
    """Load a specific or filtered sample row from the NSL-KDD dataset.

    Args:
        index: Exact row index to load. If None and label_type is specified,
               picks the first matching row or random sample.
        split: 'train' or 'test'.
        label_type: 'normal', 'attack', or None (any).
        data_dir: Path to dataset folder.

    Returns:
        Tuple of (row_as_dataframe, binary_label, original_label_string).
    """
    train_df, test_df = load_nsl_kdd(data_dir=data_dir)
    df = test_df if split == "test" else train_df

    if label_type == "normal":
        filtered_df = df[df["target"] == 0]
    elif label_type == "attack":
        filtered_df = df[df["target"] == 1]
    else:
        filtered_df = df

    if filtered_df.empty:
        raise ValueError(f"No records found with split='{split}' and label_type='{label_type}'.")

    if index is not None:
        if index not in df.index:
            raise IndexError(f"Index {index} out of bounds for {split} set (size {len(df)}).")
        row = df.loc[[index]]
    else:
        row = filtered_df.iloc[[0]]

    target = int(row["target"].values[0])
    raw_label = str(row["label"].values[0])
    return row, target, raw_label


def main() -> None:
    """Execute Phase 1 & 2 data loading, training, evaluation, and test prediction."""
    print("=========================================================")
    print("PHASE 1: PROJECT SETUP AND DATA LOADING")
    print("=========================================================")
    train_df, test_df = load_nsl_kdd(data_dir="data", download_if_missing=True)

    print(f"Train Dataset Shape: {train_df.shape}")
    print(f"Test Dataset Shape:  {test_df.shape}")
    print("\nTrain Class Distribution:")
    train_counts = train_df["target"].value_counts()
    print(f"  Normal (0): {train_counts.get(0, 0)} ({train_counts.get(0, 0)/len(train_df)*100:.2f}%)")
    print(f"  Attack (1): {train_counts.get(1, 0)} ({train_counts.get(1, 0)/len(train_df)*100:.2f}%)")

    print("\nTest Class Distribution:")
    test_counts = test_df["target"].value_counts()
    print(f"  Normal (0): {test_counts.get(0, 0)} ({test_counts.get(0, 0)/len(test_df)*100:.2f}%)")
    print(f"  Attack (1): {test_counts.get(1, 0)} ({test_counts.get(1, 0)/len(test_df)*100:.2f}%)")

    # Train and evaluate models
    train_and_evaluate_models(train_df, test_df, models_dir="models")

    # Test inference functions
    print("\nTesting inference functions...")
    sample_normal, target_n, label_n = load_sample_row(split="test", label_type="normal")
    prob_normal = get_attack_probability(sample_normal)
    print(f"Sample Normal [raw label: '{label_n}', target: {target_n}] -> P(Attack): {prob_normal:.4f}")

    sample_attack, target_a, label_a = load_sample_row(split="test", label_type="attack")
    prob_attack = get_attack_probability(sample_attack)
    print(f"Sample Attack [raw label: '{label_a}', target: {target_a}] -> P(Attack): {prob_attack:.4f}")

    assert 0.0 <= prob_normal <= 1.0, "Normal probability must be in [0, 1]"
    assert 0.0 <= prob_attack <= 1.0, "Attack probability must be in [0, 1]"
    print("\n[SUCCESS] Phase 1 and Phase 2 criteria verified successfully!")


if __name__ == "__main__":
    main()
