from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from model import load_model, predict_human_probability


@dataclass
class Sample:
    name: str
    features: dict


def make_bot_sample(rng: random.Random) -> Sample:
    features = {
        "avg_mouse_speed": rng.uniform(1.6, 3.5),
        "mouse_path_entropy": rng.uniform(0.00, 0.12),
        "click_delay": rng.uniform(0.00, 0.08),
        "task_completion_time": rng.uniform(0.15, 0.75),
        "idle_time": rng.uniform(0.00, 0.02),
        "micro_jitter_variance": rng.uniform(0.00, 0.02),
        "acceleration_curve": rng.uniform(0.00, 0.12),
        "curvature_variance": rng.uniform(0.00, 0.02),
        "overshoot_correction_ratio": rng.uniform(0.00, 0.03),
        "timing_entropy": rng.uniform(0.00, 0.10),
    }
    return Sample(name="bot_like", features=features)


def make_human_sample(rng: random.Random) -> Sample:
    features = {
        "avg_mouse_speed": rng.uniform(0.3, 1.7),
        "mouse_path_entropy": rng.uniform(0.25, 0.95),
        "click_delay": rng.uniform(0.15, 1.8),
        "task_completion_time": rng.uniform(1.2, 6.5),
        "idle_time": rng.uniform(0.05, 1.4),
        "micro_jitter_variance": rng.uniform(0.05, 2.2),
        "acceleration_curve": rng.uniform(0.2, 8.0),
        "curvature_variance": rng.uniform(0.02, 1.8),
        "overshoot_correction_ratio": rng.uniform(0.01, 0.45),
        "timing_entropy": rng.uniform(0.2, 0.95),
    }
    return Sample(name="human_like", features=features)


def summarize(label: str, probs: list[float]) -> None:
    if not probs:
        return
    arr = np.array(probs, dtype=float)
    print(
        f"{label}: n={len(probs)} mean={arr.mean():.3f} std={arr.std(ddof=1) if len(probs) > 1 else 0:.3f} min={arr.min():.3f} max={arr.max():.3f}"
    )


def main() -> int:
    payload = load_model()
    rng = random.Random(42)

    bot_probs = []
    human_probs = []

    for _ in range(30):
        bot = make_bot_sample(rng)
        human = make_human_sample(rng)
        bot_probs.append(predict_human_probability(payload, bot.features))
        human_probs.append(predict_human_probability(payload, human.features))

    print("\n=== SmartCAPTCHA Bot Simulation ===")
    print("This script generates synthetic bot-like vs human-like feature vectors and compares model confidence.\n")

    summarize("bot_like P(human)", bot_probs)
    summarize("human_like P(human)", human_probs)

    print("\nExample predictions:")
    example_bot = make_bot_sample(rng)
    example_human = make_human_sample(rng)
    print(f"bot_like  P(human)={predict_human_probability(payload, example_bot.features):.3f} features={example_bot.features}")
    print(f"human_like P(human)={predict_human_probability(payload, example_human.features):.3f} features={example_human.features}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
