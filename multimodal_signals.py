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

from rules import evaluate_condition, time_in_window


DEFAULT_POSTURE_CORRELATION = {
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

DEFAULT_CRITICAL_POSTURES = {
    "arching_back": {"pain": 0.92},
    "clenched_fists": {"pain": 0.85},
    "arms_spread": {"startle": 0.95},
    "rooting": {"hungry": 0.88},
    "sucking_fingers": {"hungry": 0.80},
    "rubbing_eyes": {"sleepy": 0.85},
    "yawning": {"sleepy": 0.90},
    "reaching_up": {"lonely": 0.85},
}

DEFAULT_CONTEXT_RULES = [
    {
        "name": "hours_since_feed",
        "rules": [
            {"condition": "> 2.5", "causes": {"hungry": 0.6}},
            {"condition": "< 0.5", "causes": {"burp": 0.3, "hungry": -0.3}},
        ],
    },
    {
        "name": "hours_since_sleep",
        "rules": [{"condition": "> 3.5", "causes": {"sleepy": 0.5}}],
    },
    {
        "name": "age_days",
        "rules": [{
            "condition": "> 21",
            "time_window": "16:00-20:00",
            "causes": {"pain": 0.2},
        }],
    },
]

DEFAULT_ENV_SENSORS = {
    "room_temp": [
        {"condition": "> 26", "causes": {"temperature": 0.70}},
        {"condition": "< 20", "causes": {"temperature": 0.65}},
    ],
    "ambient_noise_db": [
        {"condition": "> 80", "causes": {"startle": 0.55}},
    ],
}

DEFAULT_WEIGHTS = {"audio": 0.40, "vision": 0.30, "context": 0.20, "env": 0.10}
DEFAULT_AGREEMENT_BOOST = {"two_sources": 0.15, "three_sources": 0.25}
DEFAULT_CRITICAL_OVERRIDE = {
    "enabled": True,
    "min_vision_confidence": 0.0,
    "very_strong_confidence": 0.90,
    "conditions": [
        "vision_confidence_greater_than_audio",
        "context_disagrees_with_audio",
        "vision_confidence_very_strong",
    ],
}
DEFAULT_ENV_OVERRIDE = {
    "enabled": True,
    "causes": ["temperature"],
    "min_env_confidence": 0.65,
}


def profile_values(profile) -> dict:
    """Pull the generic time-based context fields off a profile object."""
    if profile is None:
        return {}
    values = {}
    if hasattr(profile, "age_days"):
        values["age_days"] = profile.age_days
    for field_name in ("hours_since_feed", "hours_since_sleep"):
        fn = getattr(profile, field_name, None)
        if callable(fn):
            try:
                values[field_name] = fn()
            except Exception:
                pass
    # Legacy aliases kept for custom rules written before the YAML schema.
    values.setdefault("hours_feed", values.get("hours_since_feed"))
    values.setdefault("hours_sleep", values.get("hours_since_sleep"))
    values.setdefault("age", values.get("age_days"))
    return values


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
    CRITICAL_POSTURES = DEFAULT_CRITICAL_POSTURES
    POSTURE_CORRELATION = DEFAULT_POSTURE_CORRELATION

    def __init__(self, posture_correlation=None, critical_postures=None,
                 critical_min_weight=0.85, critical_boost=0.10,
                 max_confidence=0.85, critical_max_confidence=0.97):
        self.posture_correlation = (
            posture_correlation
            if posture_correlation is not None
            else self.POSTURE_CORRELATION
        )
        self.critical_postures = (
            critical_postures
            if critical_postures is not None
            else self.CRITICAL_POSTURES
        )
        self.critical_min_weight = critical_min_weight
        self.critical_boost = critical_boost
        self.max_confidence = max_confidence
        self.critical_max_confidence = critical_max_confidence

    def extract(self, postures, face_state=None):
        all_postures = list(postures)
        if face_state and face_state != "normal":
            all_postures.append(f"face_{face_state}")

        scores = {}
        critical_hits = []
        for posture in all_postures:
            corr = self.posture_correlation.get(posture, {})
            is_critical = posture in self.critical_postures
            for cause, weight in corr.items():
                scores[cause] = scores.get(cause, 0) + weight
                if is_critical:
                    critical_hits.append((posture, cause, weight))

        if not scores:
            return Signal("posture", "unknown", 0.15, "vision",
                          {"postures": all_postures, "critical": False})

        best = max(scores, key=scores.get)
        total = sum(scores.values())
        base_conf = min(
            scores[best] / max(total, 1) + 0.15, self.max_confidence
        )

        is_critical = False
        for posture, cause, weight in critical_hits:
            if cause == best and weight >= self.critical_min_weight:
                base_conf = min(
                    base_conf + self.critical_boost,
                    self.critical_max_confidence,
                )
                is_critical = True

        return Signal("posture", best, round(base_conf, 3), "vision",
                      {"postures": all_postures, "all_scores": scores,
                       "critical": is_critical, "critical_hits": critical_hits})


class ContextSignalExtractor:
    def __init__(self, fields=None):
        self.fields = fields if fields is not None else DEFAULT_CONTEXT_RULES

    def extract(self, profile=None, values=None, now=None):
        ctx = profile_values(profile)
        if values:
            ctx.update(values)
        ctx.setdefault("time_of_day", time.strftime("%H:%M", now or time.localtime()))

        scores = {}
        for field_cfg in self.fields:
            name = field_cfg.get("name")
            value = ctx.get(name)
            if value is None:
                continue
            for rule in field_cfg.get("rules", []):
                window = rule.get("time_window")
                if window and not time_in_window(window, now):
                    continue
                if not evaluate_condition(rule.get("condition"), value, now):
                    continue
                for cause, weight in (rule.get("causes") or {}).items():
                    scores[cause] = scores.get(cause, 0.0) + float(weight)

        raw = {"values": ctx}
        if "hours_since_feed" in ctx:
            raw["hours_feed"] = ctx["hours_since_feed"]
        if "hours_since_sleep" in ctx:
            raw["hours_sleep"] = ctx["hours_since_sleep"]
        if "age_days" in ctx:
            raw["age"] = ctx["age_days"]
        if scores:
            raw["scores"] = {k: round(v, 3) for k, v in scores.items()}

        if not scores:
            return Signal("context", "neutral", 0.1, "context", raw)

        best = max(scores, key=scores.get)
        return Signal("context",
                      best if scores.get(best, 0) > 0 else "neutral",
                      round(abs(scores[best]), 3), "context",
                      raw)


class EnvSignalExtractor:
    def __init__(self, sensors=None):
        self.sensors = sensors if sensors is not None else DEFAULT_ENV_SENSORS

    def extract(self, room_temp=None, humidity=None, ambient_noise_db=None,
                values=None, now=None):
        sensor_values = {
            "room_temp": room_temp,
            "humidity": humidity,
            "ambient_noise_db": ambient_noise_db,
        }
        if values:
            sensor_values.update(values)

        scores = {}
        for sensor, rules in self.sensors.items():
            value = sensor_values.get(sensor)
            if value is None:
                continue
            if isinstance(rules, dict):
                rules = rules.get("rules", [])
            for rule in rules:
                if evaluate_condition(rule.get("condition"), value, now):
                    for cause, weight in (rule.get("causes") or {}).items():
                        scores[cause] = scores.get(cause, 0.0) + float(weight)

        raw = {
            "temp": room_temp,
            "noise_db": ambient_noise_db,
            "values": sensor_values,
        }
        if not scores:
            return Signal("env", "normal", 0.05, "env", raw)

        raw["scores"] = {k: round(v, 3) for k, v in scores.items()}
        best = max(scores, key=scores.get)
        return Signal("env", best, scores[best], "env", raw)


class MultimodalFuser:
    DEFAULT_WEIGHTS = DEFAULT_WEIGHTS

    def __init__(self, weights=None, agreement_boost=None,
                 escalate_threshold=0.15, critical_override=None,
                 env_override=None, escalate_confidence=0.80):
        self.weights = dict(weights or self.DEFAULT_WEIGHTS)
        boost = agreement_boost or DEFAULT_AGREEMENT_BOOST
        self.two_source_boost = float(boost.get("two_sources", 0.15))
        self.three_source_boost = float(boost.get("three_sources", 0.25))
        self.escalate_threshold = float(escalate_threshold)
        self.escalate_confidence = float(escalate_confidence)
        self.critical_cfg = (
            critical_override
            if critical_override is not None
            else dict(DEFAULT_CRITICAL_OVERRIDE)
        )
        self.env_cfg = (
            env_override
            if env_override is not None
            else dict(DEFAULT_ENV_OVERRIDE)
        )

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
        audio_sig = next(
            (d for d in signal_details.values() if d.get("source") == "audio"),
            {},
        )
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
            confidence = min(confidence + self.two_source_boost, 0.95)
        if len(agreeing_sources) >= 3:
            confidence = min(
                confidence + max(self.three_source_boost - self.two_source_boost, 0),
                0.98,
            )

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

        critical_conditions = set(self.critical_cfg.get("conditions", []))
        min_vision_conf = float(self.critical_cfg.get("min_vision_confidence", 0.0))
        very_strong = float(self.critical_cfg.get("very_strong_confidence", 0.90))
        if (self.critical_cfg.get("enabled", True)
                and is_critical
                and vision_cause != audio_cause
                and vision_conf >= min_vision_conf):
            # Condition 1: vision more confident than audio
            if ("vision_confidence_greater_than_audio" in critical_conditions
                    and vision_conf > audio_conf):
                best_cause = vision_cause
                confidence = min(vision_conf + 0.05, 0.95)
                critical_override = True
            # Condition 2: context doesn't support audio
            elif ("context_disagrees_with_audio" in critical_conditions
                    and context_cause not in (audio_cause, "neutral", "")):
                best_cause = vision_cause
                confidence = min(vision_conf, 0.85)
                critical_override = True
            # Condition 3: context is neutral but vision is very strong (>=0.90)
            elif ("vision_confidence_very_strong" in critical_conditions
                    and vision_conf >= max(very_strong, min_vision_conf)):
                best_cause = vision_cause
                confidence = min(vision_conf - 0.05, 0.85)
                critical_override = True

        # v3: Environment override for temperature
        env_cause = env_sig.get("value", "")
        env_conf = env_sig.get("confidence", 0)
        env_causes = set(self.env_cfg.get("causes", []))
        min_env_conf = float(self.env_cfg.get("min_env_confidence", 0.65))
        if (self.env_cfg.get("enabled", True)
                and env_cause in env_causes
                and env_conf >= min_env_conf):
            # Check if any posture supports temperature
            vision_supports_env = (
                vision_cause == env_cause
                or env_cause in vision_raw.get("all_scores", {})
            )
            if vision_supports_env or env_conf >= min_env_conf + 0.05:
                best_cause = env_cause
                confidence = min(env_conf + 0.10, 0.85)
                critical_override = True

        # Disagreement escalation
        margin = (best_score - second_score) / max(total, 1)
        escalate = (
            margin < self.escalate_threshold
            and confidence < self.escalate_confidence
        )

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
