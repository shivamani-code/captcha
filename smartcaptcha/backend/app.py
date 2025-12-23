
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model import load_model, predict_human_probability


app = FastAPI(title="SmartCAPTCHA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["null"],
    allow_origin_regex=r"^http://(localhost|127\\.0\\.0\\.1)(:\\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class VerifyRequest(BaseModel):
    avg_mouse_speed: float = Field(...)
    mouse_path_entropy: float = Field(...)
    click_delay: float = Field(...)
    task_completion_time: float = Field(...)
    idle_time: float = Field(...)
    micro_jitter_variance: float = Field(...)
    acceleration_curve: float = Field(...)
    curvature_variance: float = Field(...)
    overshoot_correction_ratio: float = Field(...)
    timing_entropy: float = Field(...)


class VerifyResponse(BaseModel):
    status: str
    confidence: float
    token: str


class ValidateTokenRequest(BaseModel):
    token: str


class ValidateTokenResponse(BaseModel):
    valid: bool


@dataclass
class TokenRecord:
    expires_at_s: float
    used: bool


MODEL_PAYLOAD: Optional[dict] = None
TOKEN_TTL_SECONDS = 120
TOKEN_STORE: dict[str, TokenRecord] = {}


def issue_token(now_s: float) -> str:
    token = secrets.token_urlsafe(24)
    TOKEN_STORE[token] = TokenRecord(expires_at_s=now_s + TOKEN_TTL_SECONDS, used=False)
    return token


def cleanup_tokens(now_s: float) -> None:
    expired = [t for t, rec in TOKEN_STORE.items() if rec.expires_at_s <= now_s or rec.used]
    for t in expired:
        TOKEN_STORE.pop(t, None)


def consume_token(token: str, now_s: float) -> bool:
    rec = TOKEN_STORE.get(token)
    if rec is None:
        return False
    if rec.used:
        return False
    if rec.expires_at_s <= now_s:
        TOKEN_STORE.pop(token, None)
        return False

    rec.used = True
    return True


@app.on_event("startup")
def _startup() -> None:
    global MODEL_PAYLOAD
    try:
        MODEL_PAYLOAD = load_model()
    except Exception:
        MODEL_PAYLOAD = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PAYLOAD is not None}


@app.post("/verify", response_model=VerifyResponse)
def verify(payload: VerifyRequest):
    if MODEL_PAYLOAD is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model and restart the API service.",
        )

    features = payload.model_dump()
    prob_human = predict_human_probability(MODEL_PAYLOAD, features)

    if prob_human >= 0.85:
        status = "human"
    elif prob_human >= 0.50:
        status = "suspicious"
    else:
        status = "bot"

    now_s = time.time()
    cleanup_tokens(now_s)

    token = ""
    if status == "human":
        token = issue_token(now_s)

    return VerifyResponse(status=status, confidence=float(prob_human), token=token)


@app.post("/validate", response_model=ValidateTokenResponse)
def validate(payload: ValidateTokenRequest):
    now_s = time.time()
    cleanup_tokens(now_s)
    valid = consume_token(payload.token, now_s)
    cleanup_tokens(now_s)
    return ValidateTokenResponse(valid=valid)
