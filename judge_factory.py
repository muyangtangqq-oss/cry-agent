"""Select the cloud judge: Jev by default, a domestic LLM on request."""
import os

from jev_client import JevClient
from llm_judge import LLMJudgeClient, PROVIDERS


def build_cloud_judge(cfg):
    name = os.environ.get(
        "CRY_AGENT_JUDGE_PROVIDER", "jev"
    ).strip().lower()
    if name in ("", "jev", "typesafe", "auto"):
        return JevClient(cfg)
    if name in PROVIDERS:
        return LLMJudgeClient(cfg, name)
    raise ValueError(
        f"unknown CRY_AGENT_JUDGE_PROVIDER={name!r}; "
        f"use jev or one of {sorted(PROVIDERS)}"
    )
