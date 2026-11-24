"""Jev cloud client — TypeSafe System One v1 protocol.

Routes, in order:
  1. ``CRY_AGENT_TS_BASE_URL`` (comma-separated list) when set
  2. Magpie gateway (``http://127.0.0.1:3425/v1``) — local proxy, no
     per-call key handling
  3. TypeSafe direct (``Config.typesafe_base_url``)

Every route speaks the same v1 payload built by ``typesafe_protocol``;
only the base URL and auth header differ.

Some Cloudflare edges return 451 for a valid key while others serve it, so
we retry each route once. If every route is blocked we fall back to the
offline mock for 45 s to keep the layered-fallback demo testable. Every
other failure is returned as an error instead of being silently mocked,
so the caller can pick another candidate.
"""
import json
import os
import time
import urllib.error
import urllib.request

from config import Config
from typesafe_protocol import build_choice_request, parse_choice_response

MAGPIE_BASE = "http://127.0.0.1:3425/v1"
DEFAULT_MODEL = "jev-latest"


class JevClient:
    tier_name = "jev"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.api_key = cfg.typesafe_api_key
        self.model = os.environ.get("CRY_AGENT_TS_MODEL", DEFAULT_MODEL)
        self.force_mock = os.environ.get("CRY_AGENT_TS_MOCK") == "1"
        self._routes = self._route_list()
        self._dead_routes = set()
        # Cloudflare picks the edge per connection, and some edges return
        # 451 while others serve the same key. Cooldown instead of latching
        # so a later edge switch can recover without a restart.
        self._blocked_until = 0.0

    def _route_list(self):
        env_routes = os.environ.get("CRY_AGENT_TS_BASE_URL", "")
        if env_routes.strip():
            raw = env_routes.split(",")
        else:
            raw = [MAGPIE_BASE, self.cfg.typesafe_base_url]

        routes = []
        for item in raw:
            base = item.strip().rstrip("/")
            if base and base not in routes:
                routes.append(base)
        if not routes:
            routes = [MAGPIE_BASE]
        return routes

    def _post(self, base_url, payload, timeout):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{base_url}/systemone",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def _call(self, payload):
        errors = []
        attempted = 0
        blocked = 0
        for base in self._routes:
            if base in self._dead_routes:
                continue
            attempted += 1
            timeout = 10 if "127.0.0.1" in base or "localhost" in base else 20
            for attempt in range(2):
                try:
                    data = self._post(base, payload, timeout)
                    return data, base, None, False
                except urllib.error.HTTPError as e:
                    detail = ""
                    try:
                        body = json.loads(e.read())
                        detail = str(body.get("title") or body.get("detail") or "")[:200]
                    except Exception:
                        pass
                    if e.code == 451:
                        if attempt == 0:
                            time.sleep(0.4)  # new connection, new edge
                            continue
                        blocked += 1
                        errors.append(f"{base}: 451 {detail}".strip())
                        break
                    if e.code >= 500:
                        self._dead_routes.add(base)
                    errors.append(f"{base}: {e.code} {detail}".strip())
                    break
                except (urllib.error.URLError, ConnectionError, TimeoutError,
                        OSError) as e:
                    # Nothing is listening on this route — don't pay again.
                    if "refused" in str(e).lower():
                        self._dead_routes.add(base)
                    errors.append(f"{base}: {e}")
                    break
                except Exception as e:
                    errors.append(f"{base}: {e}")
                    break
        all_451 = bool(attempted) and blocked == attempted
        if all_451:
            self._blocked_until = time.time() + 45
        return None, None, "; ".join(errors), all_451

    def judge(self, question: str, choices: list[str], state: dict) -> dict:
        """Ask Jev for a single-choice judgment.

        Signature is unchanged from the pre-v1 client so ``judge.py`` and
        ``pipeline.py`` keep working.
        """
        t0 = time.perf_counter()
        if self.force_mock or time.time() < self._blocked_until:
            result = self._mock_jev(question, choices, state)
            result["mock"] = True
            result["route"] = "mock"
            result["error"] = "451 cooldown"
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return result

        try:
            payload, qid = build_choice_request(
                self.model, question, choices, state
            )
        except ValueError as e:
            return self._error(str(e), t0)

        data, route, error, all_451 = self._call(payload)
        if data is not None:
            result = parse_choice_response(data, qid, choices)
            result["model"] = data.get("model", self.model)
            result["usage"] = data.get("usage", {})
            result["route"] = route
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return result

        if all_451:
            print(f"[jev] Region blocked (451) — using mock Jev ({error})", flush=True)
            result = self._mock_jev(question, choices, state)
            result["mock"] = True
            result["route"] = "mock"
            result["error"] = error
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return result

        return self._error(error or "all routes failed", t0)

    def _error(self, message, t0):
        return {
            "answer": None,
            "confidence": 0.0,
            "error": message,
            "model": "jev-error",
            "route": "none",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def _mock_jev(self, question, choices, state):
        """Offline simulation of Jev, used only when TypeSafe is blocked.

        Reads the audio cause + context and makes a more nuanced judgment,
        simulating the ~72 JevBench score vs Tachyone's ~38.
        """
        text = question.lower()
        audio_cause = state.get("audio_cause", "")
        audio_conf = state.get("audio_confidence", 0)
        ctx = state.get("context", {}) or {}

        hours_since_feed = ctx.get("hours_since_feed", 999)
        hours_since_sleep = ctx.get("hours_since_sleep", 999)
        age_days = ctx.get("age_days", 42)

        scores = {c: 0.1 for c in choices}  # base prior

        # Audio is the strongest signal
        if audio_cause in scores:
            scores[audio_cause] += audio_conf * 0.5

        # Context adjustments (what makes Jev "smarter")
        if hours_since_feed and hours_since_feed > 2.5 and "hungry" in scores:
            scores["hungry"] += 0.3
        elif hours_since_feed and hours_since_feed < 1.0:
            if "hungry" in scores:
                scores["hungry"] -= 0.2
            if "burp" in scores:
                scores["burp"] += 0.15  # could be post-feed gas

        if hours_since_sleep and hours_since_sleep > 3.5 and "sleepy" in scores:
            scores["sleepy"] += 0.25

        if (age_days and age_days > 30 and hours_since_feed
                and 0 < hours_since_feed < 1.0 and "pain" in scores):
            scores["pain"] += 0.1  # colic window

        if "hungry" in text and "hungry" in scores:
            scores["hungry"] += 0.05

        # Normalize
        total = sum(max(0.0, s) for s in scores.values()) or 1.0
        scores = {c: max(0.0, s) / total for c, s in scores.items()}

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
            "probabilities": {c: round(s, 3) for c, s in scores.items()},
            "model": "jev-mock-v0.1",
        }


def _smoke_test():
    from config import load_config

    cfg = load_config()
    client = JevClient(cfg)
    print(f"model alias: {client.model}")
    print(f"routes: {client._routes}")
    result = client.judge(
        "Fused analysis: hungry (60%). Audio: hungry (60%). "
        "Feed gap: 3.2h. What is the cause?",
        cfg.cry_causes,
        {
            "audio_cause": "hungry",
            "audio_confidence": 0.6,
            "context": {"hours_since_feed": 3.2, "hours_since_sleep": 2.0},
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _smoke_test()
