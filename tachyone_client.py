"""Tachyone local client — calls the local server."""
import json
import time
import urllib.request

from config import Config


class TachyoneClient:
    def __init__(self, cfg: Config):
        self.base_url = cfg.tachyone_base_url

    def judge(self, question: str, choices: list[str], state: dict) -> dict:
        payload = json.dumps({
            "question": question,
            "choices": choices,
            "state": state,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/systemone",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                data["latency_ms"] = round(
                    (time.perf_counter() - t0) * 1000, 2
                )
                return data
        except Exception as e:
            return {
                "answer": None,
                "confidence": 0.0,
                "error": str(e),
                "model": "tachyone-error",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            }
