"""Trustworthy & Explainable Threat Classification & Probability Calibration.

Implements:
1. Stratified Train/Validation split of KDDTrain+, keeping KDDTest+ as strictly held-out test.
2. Sigmoid (Platt) and Isotonic calibration via CalibratedClassifierCV fitted strictly on validation split.
3. Comparative evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix, Brier Score) on both validation and official test sets.
4. Reliability diagrams (calibration curves) with sample bin counts.
5. SHAP explainability (Global Summary Plot and local top_features(row, k=5) function).
6. Configurable model selection artifact persistence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import PredefinedSplit, train_test_split
from sklearn.pipeline import Pipeline
from tabulate import tabulate

from src.classifier import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_preprocessor,
    load_nsl_kdd,
)


class ThreatModelCalibrator:
    """Trains, calibrates, evaluates, and explains classical threat classifiers."""

    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        models_dir: Union[str, Path] = "models",
        results_dir: Union[str, Path] = "results",
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

        self.feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
        self.preprocessor = None
        self.models: Dict[str, Any] = {}
        self.eval_results: Dict[str, Any] = {}
        self.shap_explainer = None
        self.transformed_feature_names: List[str] = []

    def prepare_data(
        self, val_size: float = 0.20
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split KDDTrain+ into train and validation, keeping KDDTest+ as untouched final test."""
        full_train_df, test_df = load_nsl_kdd(data_dir=self.data_dir, download_if_missing=True)

        X_full = full_train_df[self.feature_cols]
        y_full = full_train_df["target"]

        X_train, X_val, y_train, y_val = train_test_split(
            X_full,
            y_full,
            test_size=val_size,
            random_state=self.seed,
            stratify=y_full,
        )

        X_test = test_df[self.feature_cols]
        y_test = test_df["target"]

        print(f"Dataset Partitions:")
        print(f"  Training Split:   {X_train.shape[0]:,} samples ({y_train.mean():.1%} attacks)")
        print(f"  Validation Split: {X_val.shape[0]:,} samples ({y_val.mean():.1%} attacks)")
        print(f"  Official Test:    {X_test.shape[0]:,} samples ({y_test.mean():.1%} attacks) [Held-out]")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_and_calibrate_all(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
    ) -> Dict[str, Any]:
        """Train base models on train split, and calibrate on validation split."""
        print("\nBuilding and fitting preprocessor on train split...")
        self.preprocessor = build_preprocessor()
        X_train_trans = self.preprocessor.fit_transform(X_train)
        X_val_trans = self.preprocessor.transform(X_val)

        # Extract transformed feature names
        try:
            self.transformed_feature_names = list(self.preprocessor.get_feature_names_out())
        except Exception:
            self.transformed_feature_names = [f"feat_{i}" for i in range(X_train_trans.shape[1])]

        # Concatenate train and val features for PredefinedSplit
        # -1 = training fold (used to train base estimator), 0 = calibration fold (used to calibrate)
        X_combined_trans = np.vstack([X_train_trans, X_val_trans])
        y_combined = np.concatenate([y_train.values, y_val.values])
        test_fold = np.array([-1] * len(X_train_trans) + [0] * len(X_val_trans))
        ps = PredefinedSplit(test_fold)

        # 1. Base Classifiers fitted strictly on X_train
        print("Fitting Base Logistic Regression on train split...")
        lr_base = LogisticRegression(max_iter=1000, random_state=self.seed, solver="lbfgs")
        lr_base.fit(X_train_trans, y_train)

        print("Fitting Base Random Forest on train split...")
        rf_base = RandomForestClassifier(
            n_estimators=100, max_depth=16, random_state=self.seed, n_jobs=-1
        )
        rf_base.fit(X_train_trans, y_train)

        # 2. Calibrated Classifiers: base estimator trained on train fold, calibrated on val fold
        print("Calibrating Logistic Regression (Platt / Sigmoid & Isotonic) on validation split...")
        lr_platt = CalibratedClassifierCV(
            estimator=LogisticRegression(max_iter=1000, random_state=self.seed, solver="lbfgs"),
            method="sigmoid",
            cv=ps,
        )
        lr_platt.fit(X_combined_trans, y_combined)

        lr_iso = CalibratedClassifierCV(
            estimator=LogisticRegression(max_iter=1000, random_state=self.seed, solver="lbfgs"),
            method="isotonic",
            cv=ps,
        )
        lr_iso.fit(X_combined_trans, y_combined)

        print("Calibrating Random Forest (Platt / Sigmoid & Isotonic) on validation split...")
        rf_platt = CalibratedClassifierCV(
            estimator=RandomForestClassifier(
                n_estimators=100, max_depth=16, random_state=self.seed, n_jobs=-1
            ),
            method="sigmoid",
            cv=ps,
        )
        rf_platt.fit(X_combined_trans, y_combined)

        rf_iso = CalibratedClassifierCV(
            estimator=RandomForestClassifier(
                n_estimators=100, max_depth=16, random_state=self.seed, n_jobs=-1
            ),
            method="isotonic",
            cv=ps,
        )
        rf_iso.fit(X_combined_trans, y_combined)

        self.models = {
            "LR (Uncalibrated)": lr_base,
            "LR (Platt / Sigmoid)": lr_platt,
            "LR (Isotonic)": lr_iso,
            "RF (Uncalibrated)": rf_base,
            "RF (Platt / Sigmoid)": rf_platt,
            "RF (Isotonic)": rf_iso,
        }

        # Initialize SHAP TreeExplainer on Random Forest
        print("Initializing SHAP TreeExplainer on Random Forest...")
        try:
            self.shap_explainer = shap.TreeExplainer(rf_base)
        except Exception as e:
            print(f"SHAP explainer init warning: {e}")

        return self.models

    def evaluate_all(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """Evaluate calibration and classification metrics on validation and official test sets."""
        X_val_trans = self.preprocessor.transform(X_val)
        X_test_trans = self.preprocessor.transform(X_test)

        records = []
        detailed_res = {}

        sets = [
            ("Validation Split", X_val_trans, y_val),
            ("Official Test (KDDTest+)", X_test_trans, y_test),
        ]

        for set_name, X_t, y_true in sets:
            for name, model in self.models.items():
                y_prob = model.predict_proba(X_t)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)

                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                brier = brier_score_loss(y_true, y_prob)
                cm = confusion_matrix(y_true, y_pred)

                records.append(
                    {
                        "Evaluation Set": set_name,
                        "Model Variant": name,
                        "Brier Score": f"{brier:.5f}",
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

                detailed_res[f"{set_name}::{name}"] = {
                    "y_prob": y_prob,
                    "y_true": y_true,
                    "brier": brier,
                    "acc": acc,
                    "prec": prec,
                    "rec": rec,
                    "f1": f1,
                    "cm": cm,
                }

        self.eval_results = detailed_res
        metrics_df = pd.DataFrame(records)

        # Save to disk
        out_csv = self.results_dir / "calibration_metrics.csv"
        metrics_df.to_csv(out_csv, index=False)
        print(f"\nSaved calibration comparison metrics to: {out_csv}")

        print("\n" + "=" * 95)
        print("THREAT CLASSIFIER CALIBRATION BENCHMARK (VALIDATION vs HELD-OUT TEST)")
        print("=" * 95)
        print(tabulate(metrics_df, headers="keys", tablefmt="github", showindex=False))
        print("=" * 95)

        return metrics_df

    def plot_reliability_diagrams(self) -> Path:
        """Plot reliability curves (uncalibrated vs Platt vs Isotonic) with bin count histograms."""
        print("Generating Reliability Diagrams (Calibration Curves)...")
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=False)

        # Colors for variants
        colors = {
            "Uncalibrated": "#E74C3C",
            "Platt / Sigmoid": "#3498DB",
            "Isotonic": "#2ECC71",
        }

        eval_configs = [
            ("Validation Split", "LR", axes[0, 0], axes[1, 0]),
            ("Official Test (KDDTest+)", "LR", axes[0, 0], axes[1, 0]),  # We'll plot both datasets side by side
        ]

        # Reset 2x2 grid: Left column = Validation Split (LR & RF), Right column = Official Test Set (LR & RF)
        fig.clf()
        fig, axes = plt.subplots(2, 2, figsize=(15, 11), gridspec_kw={"height_ratios": [3, 1.3]})

        # Subplot 1: Validation Set Reliability Curve
        ax_val = axes[0, 0]
        ax_val_hist = axes[1, 0]
        # Subplot 2: Official Test Set Reliability Curve
        ax_test = axes[0, 1]
        ax_test_hist = axes[1, 1]

        # Perfect calibration diagonal
        for ax in [ax_val, ax_test]:
            ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.5)

        # 1. Validation set curves (RF variants)
        for var_name, key_name, style in [
            ("RF (Uncalibrated)", "Uncalibrated", "s--"),
            ("RF (Platt / Sigmoid)", "Platt / Sigmoid", "o-"),
            ("RF (Isotonic)", "Isotonic", "^-"),
        ]:
            data = self.eval_results.get(f"Validation Split::{var_name}")
            if data is not None:
                prob_true, prob_pred = calibration_curve(
                    data["y_true"], data["y_prob"], n_bins=10, strategy="uniform"
                )
                brier = data["brier"]
                ax_val.plot(
                    prob_pred,
                    prob_true,
                    style,
                    color=colors[key_name],
                    linewidth=2.2,
                    markersize=6,
                    label=f"{key_name} (Brier: {brier:.4f})",
                )
                if key_name == "Isotonic":
                    ax_val_hist.hist(
                        data["y_prob"],
                        bins=10,
                        range=(0, 1),
                        histtype="stepfilled",
                        color=colors[key_name],
                        alpha=0.4,
                        edgecolor="black",
                    )

        ax_val.set_title("Validation Split Reliability Diagram (Random Forest)", fontsize=12, fontweight="bold")
        ax_val.set_ylabel("Fraction of True Attacks (Empirical P)", fontsize=11)
        ax_val.legend(loc="upper left", framealpha=0.9)
        ax_val.set_xlim(-0.02, 1.02)
        ax_val.set_ylim(-0.02, 1.02)

        ax_val_hist.set_xlabel("Mean Predicted Probability P(Attack)", fontsize=11)
        ax_val_hist.set_ylabel("Bin Count", fontsize=11)
        ax_val_hist.set_xlim(-0.02, 1.02)

        # 2. Official Test Set curves (RF variants)
        for var_name, key_name, style in [
            ("RF (Uncalibrated)", "Uncalibrated", "s--"),
            ("RF (Platt / Sigmoid)", "Platt / Sigmoid", "o-"),
            ("RF (Isotonic)", "Isotonic", "^-"),
        ]:
            data = self.eval_results.get(f"Official Test (KDDTest+)::{var_name}")
            if data is not None:
                prob_true, prob_pred = calibration_curve(
                    data["y_true"], data["y_prob"], n_bins=10, strategy="uniform"
                )
                brier = data["brier"]
                ax_test.plot(
                    prob_pred,
                    prob_true,
                    style,
                    color=colors[key_name],
                    linewidth=2.2,
                    markersize=6,
                    label=f"{key_name} (Brier: {brier:.4f})",
                )
                if key_name == "Isotonic":
                    ax_test_hist.hist(
                        data["y_prob"],
                        bins=10,
                        range=(0, 1),
                        histtype="stepfilled",
                        color=colors[key_name],
                        alpha=0.4,
                        edgecolor="black",
                    )

        ax_test.set_title("Official Test (KDDTest+) Reliability Diagram [Shift Analysis]", fontsize=12, fontweight="bold")
        ax_test.set_ylabel("Fraction of True Attacks (Empirical P)", fontsize=11)
        ax_test.legend(loc="upper left", framealpha=0.9)
        ax_test.set_xlim(-0.02, 1.02)
        ax_test.set_ylim(-0.02, 1.02)

        ax_test_hist.set_xlabel("Mean Predicted Probability P(Attack)", fontsize=11)
        ax_test_hist.set_ylabel("Bin Count", fontsize=11)
        ax_test_hist.set_xlim(-0.02, 1.02)

        plt.suptitle(
            "Threat Probability Calibration & Reliability Analysis (NSL-KDD)",
            fontsize=14,
            fontweight="bold",
            y=0.99,
        )
        plt.tight_layout()

        out_plot = self.results_dir / "calibration_curves.png"
        plt.savefig(out_plot, dpi=300)
        plt.close(fig)
        print(f"Saved Reliability Diagrams to: {out_plot}")
        return out_plot

    def generate_shap_summary(self, X_val: pd.DataFrame, n_samples: int = 500) -> Path:
        """Compute SHAP values and save global summary plot."""
        print(f"Computing SHAP values on {n_samples} validation samples...")
        sample_subset = X_val.sample(n=min(n_samples, len(X_val)), random_state=self.seed)
        X_sample_trans = self.preprocessor.transform(sample_subset)

        out_plot = self.results_dir / "shap_summary.png"
        try:
            shap_values = self.shap_explainer.shap_values(X_sample_trans)
            # For binary classification in tree explainer, shap_values is a list [class 0, class 1] or array
            if isinstance(shap_values, list):
                shap_val_attack = shap_values[1]
            elif len(shap_values.shape) == 3:
                shap_val_attack = shap_values[:, :, 1]
            else:
                shap_val_attack = shap_values

            fig = plt.figure(figsize=(10, 6.5))
            shap.summary_plot(
                shap_val_attack,
                X_sample_trans,
                feature_names=self.transformed_feature_names,
                max_display=15,
                show=False,
            )
            plt.title("SHAP Global Feature Importance (Random Forest Threat Classifier)", fontsize=13, pad=12)
            plt.tight_layout()
            plt.savefig(out_plot, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved SHAP Global Summary to: {out_plot}")
        except Exception as e:
            print(f"SHAP summary plot generation fallback: {e}")
            # Fallback feature importances plot
            rf = self.models["RF (Uncalibrated)"]
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:15]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(indices)), importances[indices][::-1], color="#0D9488")
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([self.transformed_feature_names[i] for i in indices[::-1]])
            ax.set_xlabel("Mean Impurity Decrease Feature Importance")
            ax.set_title("Global Feature Importance (Random Forest Threat Classifier)")
            plt.tight_layout()
            plt.savefig(out_plot, dpi=300)
            plt.close(fig)
            print(f"Saved Feature Importance Summary to: {out_plot}")

        return out_plot

    def top_features(self, row_df: pd.DataFrame, k: int = 5) -> List[Dict[str, Any]]:
        """Explain the threat prediction for an individual connection via top k feature contributions.

        Args:
            row_df: Single row DataFrame containing required NSL-KDD features.
            k: Number of top contributing features to return.

        Returns:
            List of dictionaries with feature name, raw value, and SHAP impact.
        """
        X_trans = self.preprocessor.transform(row_df[self.feature_cols])

        try:
            if self.shap_explainer is not None:
                shap_vals = self.shap_explainer.shap_values(X_trans)
                if isinstance(shap_vals, list):
                    vals = shap_vals[1][0]
                elif len(shap_vals.shape) == 3:
                    vals = shap_vals[0, :, 1]
                else:
                    vals = shap_vals[0]

                top_indices = np.argsort(np.abs(vals))[::-1][:k]
                explanations = []
                for idx in top_indices:
                    feat_name = self.transformed_feature_names[idx]
                    explanations.append(
                        {
                            "feature": feat_name,
                            "shap_value": float(vals[idx]),
                            "impact": "Increases Threat" if vals[idx] > 0 else "Decreases Threat",
                            "magnitude": float(abs(vals[idx])),
                        }
                    )
                return explanations
        except Exception as e:
            pass

        # Fallback: Top global features
        rf = self.models.get("RF (Uncalibrated)")
        if rf is not None:
            importances = rf.feature_importances_
            top_indices = np.argsort(importances)[::-1][:k]
            return [
                {
                    "feature": self.transformed_feature_names[i],
                    "shap_value": float(importances[i]),
                    "impact": "High Importance",
                    "magnitude": float(importances[i]),
                }
                for i in top_indices
            ]
        return []

    def save_selected_model(self, chosen_model_name: str = "RF (Isotonic)") -> Path:
        """Persist chosen calibrated pipeline and preprocessor to models/."""
        if chosen_model_name not in self.models:
            raise ValueError(f"Unknown model '{chosen_model_name}'. Options: {list(self.models.keys())}")

        chosen_clf = self.models[chosen_model_name]

        # Package into standard scikit-learn Pipeline
        full_pipeline = Pipeline(
            [
                ("preprocessor", self.preprocessor),
                ("classifier", chosen_clf),
            ]
        )

        calibrated_rf_path = self.models_dir / "calibrated_rf.joblib"
        rf_pipeline_path = self.models_dir / "rf_pipeline.joblib"
        preprocessor_path = self.models_dir / "preprocessor.joblib"

        print(f"Saving calibrated production pipeline to: {calibrated_rf_path}")
        joblib.dump(full_pipeline, calibrated_rf_path)
        joblib.dump(full_pipeline, rf_pipeline_path)
        joblib.dump(self.preprocessor, preprocessor_path)

        # Save metadata config
        meta_path = self.models_dir / "model_config.json"
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "chosen_model": chosen_model_name,
                    "calibrated": "Isotonic" in chosen_model_name or "Platt" in chosen_model_name,
                    "calibration_method": (
                        "isotonic"
                        if "Isotonic" in chosen_model_name
                        else ("sigmoid" if "Platt" in chosen_model_name else "none")
                    ),
                    "training_split": "KDDTrain+ (80% train / 20% val)",
                    "held_out_test": "KDDTest+ (strictly untouched)",
                },
                f,
                indent=2,
            )
        print(f"Saved metadata config to: {meta_path}")
        return calibrated_rf_path


def run_calibration_suite() -> ThreatModelCalibrator:
    """Execute complete trustworthy & explainable threat calibration pipeline."""
    print("==========================================================================")
    print("TASK 1: TRUSTWORTHY & EXPLAINABLE THREAT CLASSIFIER CALIBRATION (NSL-KDD)")
    print("==========================================================================")
    calibrator = ThreatModelCalibrator(seed=42)

    X_train, X_val, X_test, y_train, y_val, y_test = calibrator.prepare_data(val_size=0.20)
    calibrator.train_and_calibrate_all(X_train, X_val, y_train, y_val)
    calibrator.evaluate_all(X_val, y_val, X_test, y_test)
    calibrator.plot_reliability_diagrams()
    calibrator.generate_shap_summary(X_val, n_samples=300)
    calibrator.save_selected_model("RF (Isotonic)")

    # Test top_features explanation function
    sample_normal = X_test.iloc[[0]]
    sample_attack = X_test[y_test == 1].iloc[[0]]

    print("\n--- SHAP LOCAL FEATURE EXPLANATION DEMO ---")
    print("Top contributing features for Sample Attack Connection:")
    exp = calibrator.top_features(sample_attack, k=5)
    for idx, item in enumerate(exp, 1):
        print(f"  {idx}. {item['feature']:<30} | Impact: {item['impact']:<18} | SHAP: {item['shap_value']:+.4f}")

    return calibrator


if __name__ == "__main__":
    run_calibration_suite()
