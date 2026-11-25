"""Layered judge: audio → Tachyone → Jev fallback.

Architecture:
  audio → CryAudioClassifier
    ├─ confidence ≥ 0.80 → output directly
    └─ confidence < 0.80 → Tachyone (fuse with context)
         ├─ Tachyone ≥ 0.70 → output
         └─ Tachyone < 0.70 → escalate to the cloud judge
"""
import time
from dataclasses import dataclass
from typing import Optional

from config import Config
from tachyone_client import TachyoneClient
from judge_factory import build_cloud_judge
from baby_profile import BabyProfile


@dataclass
class Judgment:
    cause: str
    confidence: float
    model: str
    audio_confidence: float
    tachyone_confidence: Optional[float]
    jev_confidence: Optional[float]
    latency_ms: float
    escalated: bool
    alternatives: list[dict]
    state: dict
    judge_model: str = ""


class LayeredJudge:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tachyone = TachyoneClient(cfg)
        self.jev = build_cloud_judge(cfg)
        self.cloud_tier = getattr(self.jev, "tier_name", "jev")

    def judge(self, signals: dict, profile: BabyProfile) -> Judgment:
        causes = self.cfg.cry_causes
        ctx = profile.to_context()

        audio_cause = signals.get("audio_top_cause", "unknown")
        audio_conf = signals.get("audio_confidence", 0.0)

        # Tier 0: Audio is highly confident
        if audio_conf >= 0.80:
            return Judgment(
                cause=audio_cause, confidence=audio_conf,
                model="audio", audio_confidence=audio_conf,
                tachyone_confidence=None, jev_confidence=None,
                latency_ms=0, escalated=False,
                alternatives=signals.get("audio_probabilities", {}),
                state={"signals": signals, "context": ctx},
            )

        # Audio uncertain — build question for Tachyone/Jev
        question = self._build_question(audio_cause, audio_conf, signals, ctx)

        # Tier 1: Tachyone
        t_result = self.tachyone.judge(question, causes, {
            "audio_cause": audio_cause,
            "audio_confidence": audio_conf,
            "context": ctx,
        })
        t_conf = t_result.get("confidence", 0.0)
        t_answer = t_result.get("answer")

        # Decision point
        if t_conf >= self.cfg.tachyone_high:
            return Judgment(
                cause=t_answer, confidence=t_conf,
                model="tachyone", audio_confidence=audio_conf,
                tachyone_confidence=t_conf, jev_confidence=None,
                latency_ms=t_result.get("latency_ms", 0),
                escalated=False, alternatives=t_result.get("alternatives", []),
                state={"signals": signals, "context": ctx},
                judge_model=t_result.get("model", ""),
            )

        # Tier 2: Jev (either Tachyone uncertain or audio very low)
        j_result = self.jev.judge(question, causes, {
            "audio_cause": audio_cause,
            "audio_confidence": audio_conf,
            "context": ctx,
            "misjudgment_hints": ctx.get("misjudgment_hints", []),
        })
        jev_conf = j_result.get("confidence", 0.0)
        j_answer = j_result.get("answer")

        # Tiered arbitration: Jev is the top tier and is far better
        # calibrated than the heuristic engines, so a confident Jev answer
        # wins even when Tachyone reports a larger raw number.
        if j_answer and jev_conf > 0:
            final_cause, final_conf, model = (
                j_answer, jev_conf, self.cloud_tier
            )
            judge_model = j_result.get("model", "")
            alternatives = j_result.get("alternatives", [])
            # Jev confirming a more confident lower tier is agreement, not
            # a reason to downgrade the reported certainty.
            if j_answer == audio_cause and audio_conf > final_conf:
                final_conf = audio_conf
        elif t_answer and t_conf > 0:
            final_cause, final_conf, model = t_answer, t_conf, "tachyone"
            judge_model = t_result.get("model", "")
            alternatives = t_result.get("alternatives", [])
        else:
            final_cause, final_conf, model = audio_cause, audio_conf, "audio"
            alternatives = []

        return Judgment(
            cause=final_cause or "unknown",
            confidence=round(final_conf, 3),
            model=model,
            audio_confidence=round(audio_conf, 3),
            tachyone_confidence=round(t_conf, 3),
            jev_confidence=round(jev_conf, 3),
            latency_ms=round(
                t_result.get("latency_ms", 0) + j_result.get("latency_ms", 0), 2
            ),
            escalated=True,
            alternatives=alternatives,
            state={"signals": signals, "context": ctx},
            judge_model=judge_model,
        )

    def _build_question(self, audio_cause, audio_conf, signals, ctx):
        return (
            f"Audio classifier detected '{audio_cause}' with {audio_conf:.0%} confidence. "
            f"Hours since feed: {ctx.get('hours_since_feed', '?')}. "
            f"Hours since sleep: {ctx.get('hours_since_sleep', '?')}. "
            f"Baby age: {ctx.get('age_days', '?')} days. "
            f"What is the most likely cause of crying?"
        )
