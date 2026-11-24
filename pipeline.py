"""Full end-to-end pipeline: multimodal fusion + layered judge.

Connects:
  audio_inference → multimodal_signals (fusion) → judge (layered)
  
Fusion output feeds into the judge:
  - fused confidence ≥ 0.80 → direct output
  - 0.40-0.80 → Tachyone
  - < 0.40 → Jev
  - escalate flag → always goes to Tachyone/Jev
"""
import time
from dataclasses import dataclass

from config import Config
from multimodal_signals import Signal
from scenario_engine import Scenario, load_scenario
from tachyone_client import TachyoneClient
from jev_client import JevClient
from baby_profile import BabyProfile


@dataclass
class PipelineResult:
    cause: str
    confidence: float
    model: str
    fused_cause: str
    fused_confidence: float
    audio_cause: str
    audio_confidence: float
    vision_cause: str
    vision_confidence: float
    escalated: bool
    critical_override: bool
    latency_ms: float
    per_signal: dict
    all_scores: dict
    scenario: str = "baby_cry"


class CryAgentPipeline:
    """Full multimodal → layered judge pipeline."""

    def __init__(self, cfg: Config, scenario=None):
        self.cfg = cfg
        if isinstance(scenario, Scenario):
            self.scenario = scenario
        else:
            self.scenario = load_scenario(scenario or cfg.scenario)

        self.audio_clf = None
        self.vision_ext = self.scenario.build_vision_extractor()
        self.context_ext = self.scenario.build_context_extractor()
        self.env_ext = self.scenario.build_env_extractor()
        self.fuser = self.scenario.build_fuser()
        self.direct_threshold = self.scenario.threshold("direct_output", 0.80)
        self.tachyone_high = self.scenario.threshold("tachyone_high", 0.70)
        self.tachyone_low = self.scenario.threshold("tachyone_low", 0.40)
        self.tachyone = TachyoneClient(cfg)
        self.jev = JevClient(cfg)

    def _audio(self):
        if self.audio_clf is None:
            self.audio_clf = self.scenario.build_audio_classifier()
        return self.audio_clf

    def run(
        self,
        audio_path: str = None,
        audio_array=None,
        sr: int = None,
        postures: list[str] = None,
        face_state: str = None,
        room_temp: float = None,
        ambient_noise_db: float = None,
        profile: BabyProfile = None,
        context_values: dict = None,
        env_values: dict = None,
    ) -> PipelineResult:
        """Run the full pipeline.

        Args:
            audio_path: path to .wav file (or None for audio-less)
            postures: detected posture keywords from vision
            face_state: "red", "pale", or None
            room_temp: room temperature in Celsius
            ambient_noise_db: ambient noise level
            profile: baby profile with context timings
            context_values: extra context fields declared in the scenario
            env_values: extra sensor fields declared in the scenario
        """
        audio_clf = (
            self._audio() if (audio_path or audio_array is not None) else None
        )
        t0 = time.perf_counter()
        if profile is None:
            profile = BabyProfile(age_days=self.scenario.default_age_days)
        postures = postures or []

        # Tier 0: Extract all signals
        signals = []

        # Audio signal
        if audio_clf is not None:
            if audio_path:
                audio_result = audio_clf.classify_file(audio_path)
            else:
                audio_result = audio_clf.classify_array(
                    audio_array,
                    sr or int(self.scenario.audio_cfg.get("sample_rate", 16000)),
                )
            audio_sig = Signal(
                name="cry_classification",
                value=audio_result["cause"],
                confidence=audio_result["confidence"],
                source="audio",
                raw={"probabilities": audio_result.get("probabilities", {})},
            )
        else:
            audio_sig = Signal(
                name="cry_classification", value="unknown",
                confidence=0.0, source="audio",
            )
        signals.append(audio_sig)

        # Vision signal
        if postures or face_state:
            vision_sig = self.vision_ext.extract(postures, face_state)
        else:
            vision_sig = Signal(
                name="posture", value="unknown",
                confidence=0.0, source="vision",
            )
        signals.append(vision_sig)

        # Context signal
        context_sig = self.context_ext.extract(profile, values=context_values)
        signals.append(context_sig)

        # Env signal
        env_sig = self.env_ext.extract(
            room_temp, None, ambient_noise_db, values=env_values
        )
        signals.append(env_sig)

        # Tier 1: Fuse signals
        fused = self.fuser.fuse(signals)
        fused_cause = fused["fused_cause"]
        fused_conf = fused["fused_confidence"]
        escalate = fused.get("escalate", False)
        critical_override = fused.get("critical_override", False)

        # Tier 2: Layered judge decision
        # High confidence → direct output
        # Escalate flag or medium confidence → Tachyone
        # Low confidence → Jev
        if fused_conf >= self.direct_threshold and not escalate:
            final_cause = fused_cause
            final_conf = fused_conf
            model = "fusion"
        else:
            # Build question for Tachyone/Jev
            causes = self.scenario.classes or self.cfg.cry_causes
            question = (
                f"Fused analysis: {fused_cause} ({fused_conf:.0%}). "
                f"Audio: {audio_sig.value} ({audio_sig.confidence:.0%}). "
                f"Vision: {vision_sig.value} ({vision_sig.confidence:.0%}). "
                f"Context: {context_sig.value} ({context_sig.confidence:.0%}). "
                f"Env: {env_sig.value} ({env_sig.confidence:.0%}). "
                f"Feed gap: {profile.hours_since_feed():.1f}h. "
                f"Sleep gap: {profile.hours_since_sleep():.1f}h. "
                f"What is the cause?"
            )

            # Tachyone
            t_result = self.tachyone.judge(question, causes, {
                "fused": fused, "profile": profile.to_context(),
            })
            t_conf = t_result.get("confidence", 0.0)
            t_answer = t_result.get("answer")

            if t_conf >= self.tachyone_high and not escalate:
                final_cause = t_answer
                final_conf = t_conf
                model = "tachyone"
            else:
                # Jev
                j_result = self.jev.judge(question, causes, {
                    "fused": fused,
                    "profile": profile.to_context(),
                    "misjudgment_hints": profile.to_context().get("misjudgment_hints", []),
                })
                j_conf = j_result.get("confidence", 0.0)
                j_answer = j_result.get("answer")

                # Pick best among fusion, tachyone, jev
                candidates = [
                    ("fusion", fused_cause, fused_conf),
                    ("tachyone", t_answer, t_conf),
                    ("jev", j_answer, j_conf),
                ]
                best = max(candidates, key=lambda x: x[2] if x[1] else 0)
                final_cause = best[1] or fused_cause
                final_conf = best[2]
                model = best[0]

        latency = (time.perf_counter() - t0) * 1000

        return PipelineResult(
            cause=final_cause,
            confidence=round(final_conf, 3),
            model=model,
            fused_cause=fused_cause,
            fused_confidence=fused_conf,
            audio_cause=audio_sig.value,
            audio_confidence=audio_sig.confidence,
            vision_cause=vision_sig.value,
            vision_confidence=vision_sig.confidence,
            escalated=escalate,
            critical_override=critical_override,
            latency_ms=round(latency, 1),
            per_signal=fused["per_signal"],
            all_scores=fused["all_scores"],
            scenario=self.scenario.name,
        )
