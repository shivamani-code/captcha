from __future__ import annotations

from pathlib import Path

import joblib


def load_model(model_path: Path | None = None) -> dict:
    if model_path is None:
        model_path = Path(__file__).resolve().parent / "smartcaptcha_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run train_model.py to create it."
        )

    payload = joblib.load(model_path)
    if not isinstance(payload, dict) or "model" not in payload or "feature_columns" not in payload:
        raise ValueError("Invalid model payload. Expected keys: model, feature_columns.")

    return payload


def predict_human_probability(payload: dict, features: dict) -> float:
    model = payload["model"]
    columns = payload["feature_columns"]

    row = [[float(features.get(c, 0.0)) for c in columns]]
    proba = model.predict_proba(row)[0]

    class_index = {int(c): i for i, c in enumerate(model.classes_)}
    human_idx = class_index.get(1)
    if human_idx is None:
        return 0.0

    return float(proba[human_idx])
