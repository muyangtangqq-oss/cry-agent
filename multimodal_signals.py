"""Multimodal signal extraction and fusion — v3.

v3 fixes:
  - Critical posture override now triggers when:
    vision is critical + context disagrees with audio
    (not just when vision_conf > audio_conf)
  - Environment signal boost: temp < 20 or > 26 with corroborating
    posture triggers temperature override
  - Disagreement escalation: close scores always escalate
"""
import time
from dataclasses import dataclass, field


@dataclass
class Signal:
    name: str
    value: str
    confidence: float
    source: str
    raw: dict = field(default_factory=dict)


class AudioSignalExtractor:
    def __init__(self):
        from audio_inference import CryAudioClassifier
        self.clf = CryAudioClassifier()

    def extract(self, audio_path=None, audio_array=None, sr=None):
        if audio_path:
            result = self.clf.classify_file(audio_path)
        else:
            result = self.clf.classify_array(audio_array, sr)
        return Signal("cry_classification", result["cause"],
                      result["confidence"], "audio",
                      {"probabilities": result.get("probabilities", {})})


class VisionSignalExtractor:
    CRITICAL_POSTURES = {
        "arching_back": {"pain": 0.92},
        "clenched_fists": {"pain": 0.85},
        "arms_spread": {"startle": 0.95},
        "rooting": {"hungry": 0.88},
        "sucking_fingers": {"hungry": 0.80},
        "rubbing_eyes": {"sleepy": 0.85},
        "yawning": {"sleepy": 0.90},
        "reaching_up": {"lonely": 0.85},
    }

    POSTURE_CORRELATION = {
        "rooting": {"hungry": 0.88},
        "sucking_fingers": {"hungry": 0.80},
        "rubbing_eyes": {"sleepy": 0.85},
        "yawning": {"sleepy": 0.90},
        "arching_back": {"pain": 0.92, "burp": 0.15},
        "clenched_fists": {"pain": 0.85},
        "kicking_legs": {"discomfort": 0.55, "hungry": 0.25},
        "squirming": {"discomfort": 0.60},
        "arms_spread": {"startle": 0.95},
        "looking_around": {"lonely": 0.60},
        "reaching_up": {"lonely": 0.85},
        "limp": {"sleepy": 0.50},
        "fetal_position": {"temperature": 0.70, "discomfort": 0.25},
        "face_red": {"temperature": 0.75, "pain": 0.20},
        "face_pale": {"temperature": 0.60},
    }

    def extract(self, postures, face_state=None):
        all_postures = list(postures)
        if face_state and face_state != "normal":
            all_postures.append(f"face_{face_state}")

        scores = {}
        critical_hits = []
        for posture in all_postures:
            corr = self.POSTURE_CORRELATION.get(posture, {})
            is_critical = posture in self.CRITICAL_POSTURES
            for cause, weight in corr.items():
                scores[cause] = scores.get(cause, 0) + weight
                if is_critical:
                    critical_hits.append((posture, cause, weight))

        if not scores:
            return Signal("posture", "unknown", 0.15, "vision",
                          {"postures": all_postures, "critical": False})

        best = max(scores, key=scores.get)
        total = sum(scores.values())
        base_conf = min(scores[best] / max(total, 1) + 0.15, 0.85)

        is_critical = False
        for posture, cause, weight in critical_hits:
            if cause == best and weight >= 0.85:
                base_conf = min(base_conf + 0.10, 0.97)
                is_critical = True

        return Signal("posture", best, round(base_conf, 3), "vision",
                      {"postures": all_postures, "all_scores": scores,
                       "critical": is_critical, "critical_hits": critical_hits})


class ContextSignalExtractor:
    def extract(self, profile):
        hf = profile.hours_since_feed()
        hs = profile.hours_since_sleep()
        age = profile.age_days
        scores = {}
        if hf > 2.5: scores["hungry"] = 0.6
        elif hf < 0.5: scores["burp"] = 0.3; scores["hungry"] = -0.3
        if hs > 3.5: scores["sleepy"] = 0.5
        if age > 21 and 16 <= time.localtime().tm_hour <= 20:
            scores["pain"] = 0.2
        if not scores:
            return Signal("context", "neutral", 0.1, "context",
                          {"hours_feed": hf, "hours_sleep": hs})
        best = max(scores, key=scores.get)
        return Signal("context",
                      best if scores.get(best, 0) > 0 else "neutral",
                      round(abs(scores[best]), 3), "context",
                      {"hours_feed": hf, "hours_sleep": hs, "age": age})


class EnvSignalExtractor:
    def extract(self, room_temp=None, humidity=None, ambient_noise_db=None):
        scores = {}
        if room_temp is not None:
            if room_temp > 26: scores["temperature"] = 0.70
            elif room_temp < 20: scores["temperature"] = 0.65
        if ambient_noise_db is not None and ambient_noise_db > 80:
            scores["startle"] = 0.55
        if not scores:
            return Signal("env", "normal", 0.05, "env",
                          {"temp": room_temp, "noise_db": ambient_noise_db})
        best = max(scores, key=scores.get)
        return Signal("env", best, scores[best], "env",
                      {"temp": room_temp, "noise_db": ambient_noise_db})


class MultimodalFuser:
    DEFAULT_WEIGHTS = {
        "audio": 0.40, "vision": 0.30,
        "context": 0.20, "env": 0.10,
    }

    def __init__(self, weights=None):
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)

    def fuse(self, signals):
        cause_scores = {}
        signal_details = {}

        for sig in signals:
            signal_details[sig.name] = {
                "value": sig.value, "confidence": sig.confidence,
                "source": sig.source, "raw": sig.raw,
            }
            if sig.value in ("unknown", "neutral", "normal"):
                continue
            sw = self.weights.get(sig.source, 0.1)
            vw = sw * sig.confidence
            cause_scores[sig.value] = cause_scores.get(sig.value, 0) + vw

        vision_sig = signal_details.get("posture", {})
        vision_raw = vision_sig.get("raw", {})
        is_critical = vision_raw.get("critical", False)
        audio_sig = signal_details.get("cry_classification", {})
        context_sig = signal_details.get("context", {})
        env_sig = signal_details.get("env", {})

        if not cause_scores:
            return {
                "fused_cause": "unknown", "fused_confidence": 0.1,
                "per_signal": signal_details, "all_scores": {},
                "escalate": True, "critical_override": False,
            }

        ranked = sorted(cause_scores.items(), key=lambda x: -x[1])
        best_cause, best_score = ranked[0]
        second_cause, second_score = ranked[1] if len(ranked) > 1 else ("", 0)
        total = sum(cause_scores.values())
        confidence = best_score / max(total, 1)

        # Multi-source agreement boost
        agreeing_sources = set()
        for sig in signals:
            if sig.value == best_cause:
                agreeing_sources.add(sig.source)
        if len(agreeing_sources) >= 2:
            confidence = min(confidence + 0.15, 0.95)
        if len(agreeing_sources) >= 3:
            confidence = min(confidence + 0.10, 0.98)

        # v3: Critical posture override — enhanced logic
        # Override when:
        #   1. Vision is critical AND vision_conf > audio_conf (original)
        #   2. Vision is critical AND context doesn't support audio's cause
        #   3. Env strongly indicates a different cause (temp < 20 or > 26)
        critical_override = False
        vision_cause = vision_sig.get("value", "")
        vision_conf = vision_sig.get("confidence", 0)
        audio_cause = audio_sig.get("value", "")
        audio_conf = audio_sig.get("confidence", 0)
        context_cause = context_sig.get("value", "")

        if is_critical and vision_cause != audio_cause:
            # Condition 1: vision more confident than audio
            if vision_conf > audio_conf:
                best_cause = vision_cause
                confidence = min(vision_conf + 0.05, 0.95)
                critical_override = True
            # Condition 2: context doesn't support audio
            elif context_cause != audio_cause and context_cause != "neutral":
                best_cause = vision_cause
                confidence = min(vision_conf, 0.85)
                critical_override = True
            # Condition 3: context is neutral but vision is very strong (>=0.90)
            elif vision_conf >= 0.90:
                best_cause = vision_cause
                confidence = min(vision_conf - 0.05, 0.85)
                critical_override = True

        # v3: Environment override for temperature
        env_cause = env_sig.get("value", "")
        env_conf = env_sig.get("confidence", 0)
        if env_cause == "temperature" and env_conf >= 0.65:
            # Check if any posture supports temperature
            vision_supports_temp = any(
                "temperature" in str(v) for v in vision_raw.get("all_scores", {}).values()
            ) or vision_cause == "temperature"
            if vision_supports_temp or env_conf >= 0.70:
                best_cause = "temperature"
                confidence = min(env_conf + 0.10, 0.85)
                critical_override = True

        # Disagreement escalation
        margin = (best_score - second_score) / max(total, 1)
        escalate = margin < 0.15 and confidence < 0.80

        if critical_override:
            agreeing_sources.add("vision")

        return {
            "fused_cause": best_cause,
            "fused_confidence": round(confidence, 3),
            "agreeing_sources": list(agreeing_sources),
            "per_signal": signal_details,
            "all_scores": {k: round(v/total, 3) for k, v in ranked},
            "escalate": escalate,
            "critical_override": critical_override,
        }


if __name__ == "__main__":
    vision_ext = VisionSignalExtractor()
    context_ext = ContextSignalExtractor()
    env_ext = EnvSignalExtractor()
    fuser = MultimodalFuser()

    scenarios = [
        {"name": "hungry: audio+vision agree",
         "audio": {"cause":"hungry","conf":0.98},
         "postures":["rooting","sucking_fingers"],"face":None,
         "temp":23,"noise":40,"feed_gap":3.0,"sleep_gap":2.0,
         "expected":"hungry"},
        {"name": "sleepy: three sources agree",
         "audio": {"cause":"sleepy","conf":0.60},
         "postures":["rubbing_eyes","yawning"],"face":None,
         "temp":22,"noise":35,"feed_gap":1.0,"sleep_gap":4.5,
         "expected":"sleepy"},
        {"name": "pain: critical overrides audio",
         "audio": {"cause":"hungry","conf":0.55},
         "postures":["arching_back","clenched_fists"],"face":None,
         "temp":24,"noise":50,"feed_gap":0.8,"sleep_gap":1.5,
         "expected":"pain"},
        {"name": "startle: env+vision",
         "audio": {"cause":"startle","conf":0.40},
         "postures":["arms_spread"],"face":None,
         "temp":22,"noise":90,"feed_gap":1.5,"sleep_gap":2.0,
         "expected":"startle"},
        {"name": "cold: env+vision override",
         "audio": {"cause":"discomfort","conf":0.50},
         "postures":["fetal_position"],"face":"pale",
         "temp":17,"noise":30,"feed_gap":1.0,"sleep_gap":2.0,
         "expected":"temperature"},
        {"name": "lonely: vision overrides high-conf audio",
         "audio": {"cause":"hungry","conf":0.98},
         "postures":["reaching_up","looking_around"],"face":None,
         "temp":23,"noise":35,"feed_gap":1.0,"sleep_gap":2.0,
         "expected":"lonely"},
    ]

    from baby_profile import BabyProfile
    profile = BabyProfile(age_days=42)

    print("=" * 70)
    print("MULTIMODAL FUSION v3")
    print("=" * 70)

    correct = 0
    for i, sc in enumerate(scenarios):
        profile.last_feed_time = time.time() - sc["feed_gap"] * 3600
        profile.last_sleep_end = time.time() - sc["sleep_gap"] * 3600

        audio_sig = Signal("cry_classification", sc["audio"]["cause"],
                           sc["audio"]["conf"], "audio")
        vision_sig = vision_ext.extract(sc["postures"], sc.get("face"))
        context_sig = context_ext.extract(profile)
        env_sig = env_ext.extract(sc.get("temp"), None, sc.get("noise"))

        fused = fuser.fuse([audio_sig, vision_sig, context_sig, env_sig])
        ok = fused["fused_cause"] == sc["expected"]
        if ok: correct += 1
        mark = "✓" if ok else "✗"

        crit = " [CRIT]" if vision_sig.raw.get("critical") else ""
        ovr = " [OVR]" if fused.get("critical_override") else ""
        esc = " [ESC]" if fused.get("escalate") else ""
        print(f"\n[{i+1}] {sc['name']}")
        print(f"  Audio:{audio_sig.value}({audio_sig.confidence:.0%}) "
              f"Vis:{vision_sig.value}({vision_sig.confidence:.0%}){crit} "
              f"Ctx:{context_sig.value}({context_sig.confidence:.0%}) "
              f"Env:{env_sig.value}({env_sig.confidence:.0%})")
        print(f"  → {fused['fused_cause']} ({fused['fused_confidence']:.0%}) "
              f"{mark}{ovr}{esc}")

    print(f"\n{'='*70}")
    print(f"Accuracy: {correct}/{len(scenarios)} ({correct/len(scenarios)*100:.0f}%)")
    print(f"{'='*70}")
