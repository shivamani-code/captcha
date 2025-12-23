from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "SmartCAPTCHA backend alive"}


@app.post("/verify")
def verify(payload: dict):
    return {
        "decision": "human",
        "confidence": 0.95,
        "reason": "demo-safe human verification",
    }
