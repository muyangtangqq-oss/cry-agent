"""Configuration for cry-agent prototype."""
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    scenario: str = "baby_cry"
    typesafe_api_key: str = ""
    typesafe_base_url: str = "https://api.typesafe.ai/v1"
    tachyone_port: int = 8901
    tachyone_base_url: str = "http://127.0.0.1:8901/v1"
    tachyone_high: float = 0.70
    tachyone_low: float = 0.40
    audio_segment_seconds: float = 3.0
    audio_sample_rate: int = 16000
    cry_causes: list[str] = field(default_factory=lambda: [
        "hungry", "sleepy", "discomfort", "pain", "burp", "startle",
    ])
    default_age_days: int = 42
    default_feed_interval_hours: float = 2.5


def load_config() -> Config:
    cfg = Config()
    cfg.scenario = os.environ.get("CRY_AGENT_SCENARIO", cfg.scenario)
    import json
    p = os.path.expanduser("~/.config/magpie/providers.json")
    if os.path.exists(p):
        data = json.load(open(p))
        for prov in data.get("providers", []):
            if prov.get("id") == "typesafe":
                cfg.typesafe_api_key = prov.get("key", "")
                break
    return cfg
