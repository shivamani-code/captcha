
from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass
from math import isfinite
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from model import load_model, predict_human_probability


logger = logging.getLogger("smartcaptcha")


logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "backend alive"}


class VerifyRequest(BaseModel):
    avg_mouse_speed: float = Field(...)
    mouse_path_entropy: float = Field(...)
    click_delay: float = Field(...)
    task_completion_time: float = Field(...)
    idle_time: float = Field(...)

    @validator("avg_mouse_speed", "mouse_path_entropy", "click_delay", "task_completion_time", "idle_time")
    def _finite_number(cls, v: float):
        try:
            fv = float(v)
        except Exception as e:
            raise ValueError("value must be a number") from e
        if not isfinite(fv):
            raise ValueError("value must be finite")
        return fv


class VerifyResponse(BaseModel):
    decision: str
    confidence: float
    human_sanity: bool


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
        logger.info("Model loaded successfully")
    except Exception:
        MODEL_PAYLOAD = None
        logger.exception("Model failed to load; continuing without model")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PAYLOAD is not None}


@app.post("/verify", response_model=VerifyResponse)
def verify(payload: VerifyRequest):
    features = payload.model_dump()

    confidence = 0.0
    if MODEL_PAYLOAD is not None:
        try:
            confidence = float(predict_human_probability(MODEL_PAYLOAD, features))
        except Exception:
            confidence = 0.0
            logger.exception("Model prediction failed; defaulting confidence to 0")

    if not isfinite(confidence):
        confidence = 0.0

    human_sanity = (
        features.get("avg_mouse_speed", 0.0) > 0.15
        and features.get("mouse_path_entropy", 0.0) > 0.10
        and features.get("click_delay", 0.0) > 0.2
        and features.get("task_completion_time", 0.0) > 0.8
    )

    if human_sanity:
        decision = "human"
    elif confidence >= 0.5:
        decision = "human"
    else:
        decision = "bot"

    now_s = time.time()
    cleanup_tokens(now_s)

    logger.info("/verify: %s", {
        "features": {
            "avg_mouse_speed": features.get("avg_mouse_speed"),
            "mouse_path_entropy": features.get("mouse_path_entropy"),
            "click_delay": features.get("click_delay"),
            "task_completion_time": features.get("task_completion_time"),
            "idle_time": features.get("idle_time"),
        },
        "confidence": float(confidence),
        "human_sanity": bool(human_sanity),
        "decision": decision,
    })

    return VerifyResponse(decision=decision, confidence=float(confidence), human_sanity=bool(human_sanity))


@app.post("/validate", response_model=ValidateTokenResponse)
def validate(payload: ValidateTokenRequest):
    now_s = time.time()
    cleanup_tokens(now_s)
    valid = consume_token(payload.token, now_s)
    cleanup_tokens(now_s)
    return ValidateTokenResponse(valid=valid)
