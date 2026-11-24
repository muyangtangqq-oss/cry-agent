"""Tachyone local judgment server — lightweight, edge-deployable.

Implements a /v1/systemone-compatible API that mirrors Jev's protocol.
Uses signal-weighted scoring instead of naive keyword matching.
"""
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler


class TachyoneEngine:
    """Signal-weighted judgment engine for prototype.

    In production this would be the real 150M Tachyone model.
    For now we use structured signal scoring to simulate calibrated
    judgment with confidence.
    """

    # Each cause has a set of signal patterns with weights.
    # The engine scores each cause based on how many of its
    # expected signals are present in the input.
    CAUSE_SIGNALS = {
        "hungry": {
            "cry_pattern": {"rhythmic", "medium pitch", "short bursts", "rooting"},
            "posture": {"rooting", "sucking", "fingers", "suckle"},
            "context": {"last feed 3", "feed 2", "feed 4", "hours since feed: 3", "feed 3h"},
        },
        "sleepy": {
            "cry_pattern": {"whimpering", "low energy", "fussing low"},
            "posture": {"rubbing eyes", "yawning", "rub eyes", "yawn"},
            "context": {"last sleep 4", "sleep 3", "sleep 4", "hours since sleep: 4", "sleep 3h"},
        },
        "discomfort": {
            "cry_pattern": {"fussing", "intermittent", "intermittent fussing"},
            "posture": {"squirming", "kicking", "kicking legs", "squirm"},
            "context": {"diaper", "wet", "hot", "cold", "tight", "clothes"},
        },
        "pain": {
            "cry_pattern": {"sharp", "high pitch", "sustained", "sharp high"},
            "posture": {"arching", "arch", "clenched", "fists", "arched back"},
            "context": {"evening", "colic", "last feed 30", "reflux", "0.5"},
        },
        "burp": {
            "cry_pattern": {"fussing after", "brief cry"},
            "posture": {"squirming after feed", "arching slightly"},
            "context": {"after feed", "just fed", "gas", "burp", "spit"},
        },
        "startle": {
            "cry_pattern": {"sudden", "scream", "sudden scream"},
            "posture": {"startled", "arms spread", "startle", "jump"},
            "context": {"loud", "noise", "fell", "sudden noise"},
        },
    }

    def judge(self, question: str, choices: list[str], state: dict) -> dict:
        """Return structured judgment: {answer, confidence, alternatives}."""
        text = question.lower()

        # Also extract signals from state
        signals = state.get("signals", {})
        signal_text = json.dumps(signals).lower()
        context = state.get("context", {})
        context_text = json.dumps(context).lower()
        full_text = text + " " + signal_text + " " + context_text

        scores = {}
        for cause in choices:
            cause_def = self.CAUSE_SIGNALS.get(cause, {})
            if not cause_def:
                scores[cause] = 0.0
                continue

            hits = 0
            possible = 0
            for category, keywords in cause_def.items():
                possible += len(keywords)
                for kw in keywords:
                    if kw in full_text:
                        hits += 1

            # Coverage ratio
            scores[cause] = hits / max(possible, 1)

        # Find best
        best_cause = max(scores, key=scores.get)
        best_score = scores[best_cause]

        # Convert coverage to confidence
        if best_score == 0:
            # No signal match at all
            n = len(choices)
            scores = {c: 1.0 / n for c in choices}
            confidence = 0.12
        else:
            # Confidence based on how dominant the best cause is
            # and absolute coverage
            second = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
            dominance = best_score - second
            confidence = min(0.40 + best_score * 0.50 + dominance * 0.30, 0.95)

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        answer = ranked[0][0]
        alternatives = [
            {"answer": c, "confidence": round(s, 3)}
            for c, s in ranked[1:] if s > 0.01
        ]

        return {
            "answer": answer,
            "confidence": round(confidence, 3),
            "alternatives": alternatives[:3],
            "model": "tachyone-heuristic-v0.2",
            "latency_ms": 0,
        }


ENGINE = TachyoneEngine()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/systemone":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        question = body.get("question", "")
        choices = body.get("choices", [])
        state = body.get("state", {})

        t0 = time.perf_counter()
        result = ENGINE.judge(question, choices, state)
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        payload = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(payload))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


def run(port=8901):
    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Tachyone local server on http://127.0.0.1:{port}/v1/systemone", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8901
    run(port)
