"""Tachyone local judgment server — lightweight, edge-deployable.

Implements a /v1/systemone-compatible API that mirrors Jev's v1 protocol
(``state`` + ``questions`` → ``answers``). The legacy
``{question, choices, state}`` shape is still accepted so old clients and
saved requests keep working.
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
        context = state.get("context") or state.get("profile") or {}
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

        # Structured evidence is worth more than keyword coverage: the
        # caller already gives us calibrated audio and fusion outputs.
        audio_cause = state.get("audio_cause")
        audio_conf = self._num(state.get("audio_confidence")) or 0.0
        if audio_cause in scores:
            scores[audio_cause] += 0.9 * audio_conf

        fused = state.get("fused") or {}
        fused_cause = fused.get("fused_cause")
        fused_conf = self._num(fused.get("fused_confidence")) or 0.0
        if fused_cause in scores:
            scores[fused_cause] += 0.9 * fused_conf

        feed = self._num(context.get("hours_since_feed"))
        sleep = self._num(context.get("hours_since_sleep"))
        age = self._num(context.get("age_days"))
        if feed is not None:
            if feed > 2.5:
                self._add(scores, "hungry", min(0.6, 0.12 * (feed - 2.0)))
            elif feed < 0.75:
                self._add(scores, "burp", 0.45)
                self._add(scores, "hungry", -0.3)
        if sleep is not None and sleep > 3.5:
            self._add(scores, "sleepy", min(0.55, 0.12 * (sleep - 3.0)))
        room_temp = self._num(context.get("room_temp"))
        if room_temp is not None and (room_temp > 26 or room_temp < 20):
            self._add(scores, "temperature", 0.6)
        noise = self._num(context.get("ambient_noise_db"))
        if noise is not None and noise > 80:
            self._add(scores, "startle", 0.55)
        if age is not None and age > 21 and feed is not None and 0 < feed < 1:
            self._add(scores, "pain", 0.2)  # evening colic window

        scores = {c: max(0.0, s) for c, s in scores.items()}

        # Find best
        best_cause = max(scores, key=scores.get)
        best_score = scores[best_cause]

        # Convert coverage to confidence
        if best_score <= 0:
            # No signal match at all
            n = len(choices)
            scores = {c: 1.0 / n for c in choices}
            confidence = 0.12
        else:
            # Confidence = share of the probability mass, tempered by how
            # much evidence exists at all. A 35%-confident audio label plus
            # nothing else must not become a 90%-confident local verdict.
            second = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
            total = sum(scores.values())
            top_prob = best_score / total
            dominance = (best_score - second) / total
            evidence_strength = min(1.0, best_score / 0.9)
            confidence = min(
                0.95,
                top_prob * (0.35 + 0.60 * evidence_strength) + 0.10 * dominance,
            )

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        answer = ranked[0][0]
        total = sum(max(0.0, s) for s in scores.values()) or 1.0
        probabilities = {
            c: round(max(0.0, s) / total, 3) for c, s in scores.items()
        }
        alternatives = [
            {"answer": c, "confidence": probabilities[c]}
            for c, s in ranked[1:] if s > 0.01
        ]

        return {
            "answer": answer,
            "confidence": round(confidence, 3),
            "alternatives": alternatives[:3],
            "probabilities": probabilities,
            "model": "tachyone-heuristic-v0.2",
            "latency_ms": 0,
        }

    @staticmethod
    def _num(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _add(scores, cause, weight):
        if cause in scores:
            scores[cause] += weight


ENGINE = TachyoneEngine()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/systemone":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        t0 = time.perf_counter()

        if "questions" in body:
            questions = body.get("questions") or {}
            if not questions:
                self.send_error(400, "questions must not be empty")
                return
            qid, question = next(iter(questions.items()))
            criteria = question.get("criteria") or {}
            if not isinstance(criteria, dict):
                self.send_error(400, "only choice questions are supported")
                return
            choices = list(criteria.keys())
            if not choices:
                self.send_error(400, "criteria must not be empty")
                return
            result = ENGINE.judge(
                question.get("instructions") or "",
                choices,
                body.get("state") or {},
            )
            payload = {
                "model": result["model"],
                "answers": {
                    qid: {
                        "type": "choice",
                        "choice": result["answer"],
                        "confidence": result["confidence"],
                        "probabilities": result["probabilities"],
                    }
                },
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            }
        else:
            # Legacy shape
            result = ENGINE.judge(
                body.get("question", ""),
                body.get("choices", []),
                body.get("state", {}),
            )
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            payload = result

        payload = json.dumps(payload).encode()
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
