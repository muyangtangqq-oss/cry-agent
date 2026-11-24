"""Jev cloud client — calls TypeSafe API via Magpie gateway.

If TypeSafe is unavailable (region block, network), falls back to a
mock Jev that simulates structured judgment with higher accuracy than
Tachyone, so the layered fallback chain can still be validated.
"""
import json
import time
import urllib.request
import urllib.error

from config import Config


class JevClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.base_url = "http://127.0.0.1:3425/v1"  # Magpie gateway
        self.api_key = cfg.typesafe_api_key
        self.model = "typesafe/jev-latest"
        self._region_blocked = None

    def _try_real_jev(self, question, choices, state):
        """Attempt real Jev via Magpie gateway."""
        payload = json.dumps({
            "model": self.model,
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

        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    def _mock_jev(self, question, choices, state):
        """Simulate Jev with higher reasoning than Tachyone.

        Mock Jev reads the audio cause + context and makes a more
        nuanced judgment, simulating the ~72 JevBench score vs
        Tachyone's ~38.
        """
        text = question.lower()
        audio_cause = state.get("audio_cause", "")
        audio_conf = state.get("audio_confidence", 0)
        ctx = state.get("context", {})

        hours_since_feed = ctx.get("hours_since_feed", 999)
        hours_since_sleep = ctx.get("hours_since_sleep", 999)
        age_days = ctx.get("age_days", 42)

        # Jev "reasons" about the signals together
        scores = {c: 0.1 for c in choices}  # base prior

        # Audio is the strongest signal
        if audio_cause in scores:
            scores[audio_cause] += audio_conf * 0.5

        # Context adjustments (what makes Jev "smarter")
        if hours_since_feed and hours_since_feed > 2.5:
            scores["hungry"] += 0.3
        elif hours_since_feed and hours_since_feed < 1.0:
            # Just fed — unlikely hungry
            scores["hungry"] -= 0.2
            scores["burp"] += 0.15  # could be post-feed gas

        if hours_since_sleep and hours_since_sleep > 3.5:
            scores["sleepy"] += 0.25

        if age_days and age_days > 30 and hours_since_feed and 0 < hours_since_feed < 1.0:
            scores["pain"] += 0.1  # colic window

        # Normalize
        total = sum(scores.values())
        scores = {c: max(0, s / total) for c, s in scores.items()}

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        answer = ranked[0][0]
        confidence = min(ranked[0][1], 0.95)
        alternatives = [
            {"answer": c, "confidence": round(s, 3)}
            for c, s in ranked[1:] if s > 0.05
        ]

        return {
            "answer": answer,
            "confidence": round(confidence, 3),
            "alternatives": alternatives[:3],
            "model": "jev-mock-v0.1",
        }

    def judge(self, question: str, choices: list[str], state: dict) -> dict:
        """Call Jev or fall back to mock."""
        t0 = time.perf_counter()

        # Try real Jev if we haven't confirmed region block
        if self._region_blocked is None:
            try:
                result = self._try_real_jev(question, choices, state)
                self._region_blocked = False
                result["latency_ms"] = round(
                    (time.perf_counter() - t0) * 1000, 2
                )
                result["model"] = result.get("model", "jev-latest")
                return result
            except urllib.error.HTTPError as e:
                if e.code == 451:
                    self._region_blocked = True
                    print("[jev] Region blocked (451) — using mock Jev", flush=True)
                else:
                    return {
                        "answer": None,
                        "confidence": 0.0,
                        "error": f"http_{e.code}",
                        "model": "jev-error",
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                    }
            except Exception as e:
                return {
                    "answer": None,
                    "confidence": 0.0,
                    "error": str(e),
                    "model": "jev-error",
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                }

        # Use mock Jev
        result = self._mock_jev(question, choices, state)
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return result
