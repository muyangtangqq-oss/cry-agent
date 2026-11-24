"""Baby profile — auto-updating personal state (Level 1 self-learning)."""
import json
import os
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional


def _sanitize(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@dataclass
class CryEvent:
    timestamp: float
    cause: str
    confidence: float
    signals: dict
    feedback: Optional[str] = None


@dataclass
class BabyProfile:
    name: str = "baby"
    age_days: int = 42

    last_feed_time: float = 0.0
    last_sleep_start: float = 0.0
    last_sleep_end: float = 0.0
    last_diaper_change: float = 0.0

    feed_interval_mean_hours: float = 2.5
    feed_interval_samples: list[float] = field(default_factory=list)

    cry_history: list[dict] = field(default_factory=list)

    signal_weights: dict[str, float] = field(default_factory=lambda: {
        "cry_pattern": 0.40,
        "posture": 0.30,
        "context": 0.15,
        "time_pattern": 0.15,
    })

    misjudgment_notes: list[str] = field(default_factory=list)

    def hours_since_feed(self) -> float:
        if not self.last_feed_time:
            return 999.0
        return (time.time() - self.last_feed_time) / 3600

    def hours_since_sleep(self) -> float:
        if not self.last_sleep_end:
            return 999.0
        return (time.time() - self.last_sleep_end) / 3600

    def record_cry(self, event: CryEvent):
        self.cry_history.append(asdict(event))
        if len(self.cry_history) > 500:
            self.cry_history = self.cry_history[-500:]

    def record_feedback(self, event_idx: int, actual_cause: str,
                        correct_boost: float = 1.05,
                        wrong_penalty: float = 0.90):
        """Adjust weights based on feedback (Level 2)."""
        if event_idx >= len(self.cry_history):
            return
        event = self.cry_history[event_idx]
        event["feedback"] = actual_cause

        if event["cause"] == actual_cause:
            for sig_key in self.signal_weights:
                self.signal_weights[sig_key] *= correct_boost
        else:
            note = f'Predicted "{event["cause"]}" but was "{actual_cause}".'
            self.misjudgment_notes.append(note)
            top_sig = max(self.signal_weights, key=self.signal_weights.get)
            self.signal_weights[top_sig] *= wrong_penalty

        total = sum(self.signal_weights.values())
        if total > 0:
            self.signal_weights = {
                k: v / total for k, v in self.signal_weights.items()
            }

    def update_feed_time(self, ts: float = None):
        self.last_feed_time = ts or time.time()
        if len(self.feed_interval_samples) > 0:
            gap = (self.last_feed_time - self.feed_interval_samples[-1]) / 3600
            if 0.5 < gap < 8:
                self.feed_interval_samples.append(gap)
                recent = self.feed_interval_samples[-20:]
                self.feed_interval_mean_hours = sum(recent) / len(recent)

    def to_context(self) -> dict:
        return {
            "age_days": self.age_days,
            "hours_since_feed": round(self.hours_since_feed(), 1),
            "hours_since_sleep": round(self.hours_since_sleep(), 1),
            "avg_feed_interval_hours": self.feed_interval_mean_hours,
            "recent_cry_count_24h": sum(
                1 for e in self.cry_history
                if time.time() - e["timestamp"] < 86400
            ),
            "signal_weights": {k: round(v, 3) for k, v in self.signal_weights.items()},
            "misjudgment_hints": self.misjudgment_notes[-5:],
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(_sanitize(asdict(self)), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BabyProfile":
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            return cls(**{k: v for k, v in data.items()
                          if k in cls.__dataclass_fields__})
        return cls()
