from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "captcha_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED = True
else:
    model = None
    MODEL_LOADED = False
    print("WARNING: captcha_model.pkl not found. Running in fallback mode.")

@app.get("/")
def root():
    return {"status": "backend alive"}

@app.post("/verify")
def verify(payload: dict):
    # If model is missing, allow human (fallback)
    if not MODEL_LOADED:
        return {
            "decision": "human",
            "confidence": 0.5,
            "mode": "fallback-no-model"
        }

    # Extract features in correct order
    features = [[
        payload["avg_mouse_speed"],
        payload["mouse_path_entropy"],
        payload["click_delay"],
        payload["task_completion_time"],
        payload["idle_time"]
    ]]

    # ML confidence (probability of human)
    confidence = float(model.predict_proba(features)[0][1])

    # Human-like sanity checks (IMPORTANT)
    human_like = (
        payload["avg_mouse_speed"] > 0.25 and
        payload["mouse_path_entropy"] > 0.15 and
        payload["click_delay"] > 0.4 and
        payload["task_completion_time"] > 1.2
    )

    # Final decision (SAFE)
    if human_like or confidence >= 0.40:
        decision = "human"
    else:
        decision = "bot"

    return {
        "decision": decision,
        "confidence": round(confidence, 3),
        "mode": "ml-enabled"
    }
