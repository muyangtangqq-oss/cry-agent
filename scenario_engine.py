"""Configuration-driven scenario engine.

One processing stack, many domains. A scenario YAML declares:
  - which signals are available and what they mean
  - posture → cause correlation maps
  - context/env rules
  - fusion weights and override policy
  - judge thresholds
  - profile defaults and self-learning knobs

Adding a new domain (elderly care, pet behavior, ...) should be a YAML file,
not a new pipeline.
"""
import os
from dataclasses import dataclass
from typing import Any

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


def list_scenarios(config_dir: str = CONFIG_DIR) -> dict[str, str]:
    """Return {scenario_name: path} for every YAML in configs/."""
    scenarios = {}
    if not os.path.isdir(config_dir):
        return scenarios
    for fname in sorted(os.listdir(config_dir)):
        if fname.endswith((".yaml", ".yml")):
            scenarios[os.path.splitext(fname)[0]] = os.path.join(config_dir, fname)
    return scenarios


def resolve_scenario_path(name_or_path: str = "baby_cry",
                          config_dir: str = CONFIG_DIR) -> str:
    """Accept a scenario name, a name with .yaml, or a filesystem path."""
    if not name_or_path:
        name_or_path = "baby_cry"
    expanded = os.path.expanduser(name_or_path)
    if os.path.isfile(expanded):
        return expanded
    if os.path.isfile(expanded + ".yaml"):
        return expanded + ".yaml"
    candidate = os.path.join(config_dir, name_or_path)
    if os.path.isfile(candidate):
        return candidate
    candidate += ".yaml"
    if os.path.isfile(candidate):
        return candidate
    available = ", ".join(list_scenarios(config_dir)) or "(none)"
    raise FileNotFoundError(
        f"Scenario '{name_or_path}' not found. Available: {available}"
    )


@dataclass
class Scenario:
    raw: dict[str, Any]
    path: str = ""

    @classmethod
    def load(cls, name_or_path: str = "baby_cry",
             config_dir: str = CONFIG_DIR) -> "Scenario":
        path = resolve_scenario_path(name_or_path, config_dir)
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Scenario {path} must contain a YAML mapping")
        return cls(raw=raw, path=path)

    # ---- top-level sections -------------------------------------------------
    @property
    def name(self) -> str:
        return self.raw.get("name") or os.path.splitext(os.path.basename(self.path))[0]

    @property
    def description(self) -> str:
        return self.raw.get("description", "")

    @property
    def signals(self) -> dict:
        return self.raw.get("signals") or {}

    @property
    def audio_cfg(self) -> dict:
        return self.signals.get("audio") or {}

    @property
    def vision_cfg(self) -> dict:
        return self.signals.get("vision") or {}

    @property
    def context_cfg(self) -> dict:
        return self.signals.get("context") or {}

    @property
    def env_cfg(self) -> dict:
        return self.signals.get("env") or {}

    @property
    def fusion_cfg(self) -> dict:
        return self.raw.get("fusion") or {}

    @property
    def judge_cfg(self) -> dict:
        return self.raw.get("judge") or {}

    @property
    def thresholds(self) -> dict:
        return self.judge_cfg.get("thresholds") or {}

    @property
    def self_learning(self) -> dict:
        return self.raw.get("self_learning") or {}

    @property
    def profile_cfg(self) -> dict:
        return self.raw.get("profile") or {}

    # ---- convenience --------------------------------------------------------
    @property
    def classes(self) -> list[str]:
        return list(self.audio_cfg.get("classes") or [])

    @property
    def default_age_days(self) -> int:
        return int(self.profile_cfg.get("default_age_days", 42))

    @property
    def audio_enabled(self) -> bool:
        return bool(self.audio_cfg.get("enabled", True))

    @property
    def vision_enabled(self) -> bool:
        return bool(self.vision_cfg.get("enabled", True))

    @property
    def context_enabled(self) -> bool:
        return bool(self.context_cfg.get("enabled", True))

    @property
    def env_enabled(self) -> bool:
        return bool(self.env_cfg.get("enabled", True))

    def threshold(self, key: str, default: float) -> float:
        return float(self.thresholds.get(key, default))

    # ---- component builders -------------------------------------------------
    def build_audio_classifier(self):
        if not self.audio_enabled:
            raise RuntimeError(f"Scenario '{self.name}' has audio disabled")
        from audio_inference import CryAudioClassifier
        model_path = self.audio_cfg.get("model_path")
        encoder_path = self.audio_cfg.get("label_encoder_path")
        model_dir = os.environ.get("CRY_AGENT_MODEL_DIR")
        if model_dir:
            model_path = os.path.join(model_dir, "cry_classifier.joblib")
            encoder_path = os.path.join(model_dir, "label_encoder.joblib")
        model_path = os.environ.get("CRY_AGENT_MODEL_PATH", model_path)
        encoder_path = os.environ.get("CRY_AGENT_LABEL_ENCODER_PATH", encoder_path)
        ood_path = self.audio_cfg.get("ood_detector_path")
        if not ood_path and model_path:
            ood_path = os.path.join(
                os.path.dirname(os.path.expanduser(model_path)),
                "ood_detector.joblib",
            )
        ood_path = os.environ.get("CRY_AGENT_OOD_PATH", ood_path)
        adapter_path = (
            self.audio_cfg.get("family_adapter")
            or os.environ.get("CRY_AGENT_FAMILY_ADAPTER")
        )
        classifier = CryAudioClassifier(
            model_path=model_path,
            label_encoder_path=encoder_path,
            segment_duration=self.audio_cfg.get("segment_duration"),
            sample_rate=self.audio_cfg.get("sample_rate"),
            ood_path=ood_path,
        )
        if adapter_path:
            classifier.load_family_adapter(
                os.path.expanduser(str(adapter_path))
            )
        return classifier

    def build_vision_extractor(self):
        from multimodal_signals import VisionSignalExtractor
        if not self.vision_enabled:
            return VisionSignalExtractor(
                posture_correlation={}, critical_postures={}
            )
        return VisionSignalExtractor(
            posture_correlation=self.vision_cfg.get("posture_correlation"),
            critical_postures=self.vision_cfg.get("critical_postures"),
        )

    def build_context_extractor(self):
        from multimodal_signals import ContextSignalExtractor
        fields = self.context_cfg.get("fields", []) if self.context_enabled else []
        return ContextSignalExtractor(fields=fields)

    def build_env_extractor(self):
        from multimodal_signals import EnvSignalExtractor
        sensors = self.env_cfg.get("sensors", {}) if self.env_enabled else {}
        return EnvSignalExtractor(sensors=sensors)

    def build_fuser(self):
        from multimodal_signals import MultimodalFuser
        fusion = self.fusion_cfg
        return MultimodalFuser(
            weights=fusion.get("weights"),
            agreement_boost=fusion.get("agreement_boost"),
            escalate_threshold=fusion.get(
                "disagreement_escalate_threshold", 0.15
            ),
            critical_override=fusion.get("critical_override"),
            env_override=fusion.get("env_override"),
            escalate_confidence=self.threshold("direct_output", 0.85),
        )

    def summary(self) -> str:
        lines = [
            f"{self.name}",
            f"  file: {self.path}",
            f"  {self.description}",
            f"  audio: {'on' if self.audio_enabled else 'off'} "
            f"({len(self.classes)} classes)",
            f"  vision: {'on' if self.vision_enabled else 'off'} "
            f"({len(self.vision_cfg.get('posture_correlation', {}))} postures)",
            f"  context: {'on' if self.context_enabled else 'off'} "
            f"({len(self.context_cfg.get('fields', []))} fields)",
            f"  env: {'on' if self.env_enabled else 'off'} "
            f"({len(self.env_cfg.get('sensors', {}))} sensors)",
            f"  thresholds: direct={self.threshold('direct_output', 0.85):.2f} "
            f"tachyone={self.threshold('tachyone_high', 0.70):.2f} "
            f"low={self.threshold('tachyone_low', 0.40):.2f}",
            f"  fusion weights: {self.fusion_cfg.get('weights', {})}",
        ]
        return "\n".join(lines)


def load_scenario(name_or_path: str = "baby_cry") -> Scenario:
    return Scenario.load(name_or_path)
