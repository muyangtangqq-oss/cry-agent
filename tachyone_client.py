"""Tachyone local client — calls the local server over v1 SystemOne."""
import json
import time
import urllib.request

from config import Config
from typesafe_protocol import build_choice_request, parse_choice_response


class TachyoneClient:
    def __init__(self, cfg: Config):
        self.base_url = cfg.tachyone_base_url

    def judge(self, question: str, choices: list[str], state: dict) -> dict:
        payload_dict, qid = build_choice_request(
            "tachyone-heuristic", question, choices, state
        )

        req = urllib.request.Request(
            f"{self.base_url}/systemone",
            data=json.dumps(payload_dict).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                result = parse_choice_response(data, qid, choices)
                result["model"] = data.get("model", "tachyone")
                result["usage"] = data.get("usage", {})
                result["latency_ms"] = round(
                    (time.perf_counter() - t0) * 1000, 2
                )
                return result
        except Exception as e:
            return {
                "answer": None,
                "confidence": 0.0,
                "error": str(e),
                "model": "tachyone-error",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            }
