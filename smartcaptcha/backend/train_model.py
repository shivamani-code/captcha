from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


REQUIRED_FEATURE_COLUMNS = [
    "avg_mouse_speed",
    "mouse_path_entropy",
    "click_delay",
    "task_completion_time",
    "idle_time",
    "micro_jitter_variance",
    "acceleration_curve",
    "curvature_variance",
    "overshoot_correction_ratio",
    "timing_entropy",
]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(
            "Dataset is empty. Add rows to dataset/behavior_data.csv before training."
        )

    missing = [c for c in (REQUIRED_FEATURE_COLUMNS + ["label"]) if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df = df.dropna(subset=REQUIRED_FEATURE_COLUMNS + ["label"]).copy()
    if df.empty:
        raise ValueError("Dataset has no usable rows after dropping missing values.")

    df["label"] = df["label"].astype(int)
    return df


def train_model(df: pd.DataFrame) -> tuple[RandomForestClassifier, dict]:
    X = df[REQUIRED_FEATURE_COLUMNS]
    y = df["label"]

    if y.nunique() < 2:
        raise ValueError(
            "Dataset must contain at least 2 classes (human=1 and bot=0) to train."
        )

    class_counts = y.value_counts().to_dict()
    min_class_count = min(class_counts.values())
    if min_class_count < 2:
        raise ValueError(
            "Each class must have at least 2 samples to train. "
            f"Current counts: {class_counts}."
        )

    # For very small datasets, train on all available rows so you can test end-to-end
    # (API wiring + inference) without being blocked by split/metrics constraints.
    if len(df) < 8:
        clf = RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
        )
        clf.fit(X, y)

        metrics = {
            "accuracy": None,
            "test_rows": 0,
            "train_rows": int(len(X)),
            "labels": sorted([int(v) for v in y.unique().tolist()]),
            "note": "Trained on all rows (dataset too small for a reliable train/test split).",
        }

        print("\n=== SmartCAPTCHA Model Training ===")
        print(f"Train rows: {metrics['train_rows']}")
        print("Test rows:  0")
        print("Accuracy:   N/A")
        print("Note: Dataset is small; add more rows for meaningful evaluation.")

        return clf, metrics

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "test_rows": int(len(X_test)),
        "train_rows": int(len(X_train)),
        "labels": sorted([int(v) for v in y.unique().tolist()]),
    }

    print("\n=== SmartCAPTCHA Model Training ===")
    print(f"Train rows: {metrics['train_rows']}")
    print(f"Test rows:  {metrics['test_rows']}")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print("\nClassification report:\n")
    print(report)

    return clf, metrics


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = repo_root / "dataset" / "behavior_data.csv"
    model_path = Path(__file__).resolve().parent / "smartcaptcha_model.joblib"

    df = load_dataset(dataset_path)
    clf, _ = train_model(df)

    payload = {
        "model": clf,
        "feature_columns": REQUIRED_FEATURE_COLUMNS,
        "label_mapping": {"bot": 0, "human": 1},
    }
    joblib.dump(payload, model_path)
    print(f"\nSaved model to: {model_path}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
