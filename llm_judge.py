"""OpenAI-compatible cloud judge — a mainland-accessible Jev alternative.

Jev's advantage is a typed, calibrated decision. A small instruct model
with JSON mode is not identical, but for this narrow task (choose one of
six causes from structured evidence) it is a practical substitute:

  deepseek  https://api.deepseek.com/v1            deepseek-chat
  qwen      https://dashscope.aliyuncs.com/compatible-mode/v1  qwen-plus
  glm       https://open.bigmodel.cn/api/paas/v4   glm-4-flash
  kimi      https://api.moonshot.cn/v1             moonshot-v1-8k
  ollama    http://127.0.0.1:11434/v1              qwen2.5:1.5b-instruct
  openai    https://api.openai.com/v1              gpt-4.1-mini

Select with ``CRY_AGENT_JUDGE_PROVIDER``. Keys are read from
``CRY_AGENT_JUDGE_API_KEY`` first, then from Magpie's providers.json.
``ollama`` needs no key and is fully offline.

Usage:
  CRY_AGENT_JUDGE_PROVIDER=deepseek python llm_judge.py
  CRY_AGENT_JUDGE_PROVIDER=ollama python llm_judge.py
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

from config import Config
from typesafe_protocol import CAUSE_CRITERIA

PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "key_id": "deepseek",
        "json_mode": True,
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "key_id": "qwen",
        "json_mode": True,
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
        "key_id": "zhipu",
        "json_mode": True,
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
        "key_id": "moonshot",
        "json_mode": True,
    },
    "ollama": {
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen2.5:1.5b-instruct",
        "key_id": None,
        "json_mode": False,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4.1-mini",
        "key_id": "openai-api",
        "json_mode": True,
    },
}


def load_provider_key(key_id):
    path = os.path.expanduser("~/.config/magpie/providers.json")
    if not key_id or not os.path.exists(path):
        return ""
    try:
        data = json.load(open(path))
    except Exception:
        return ""
    for provider in data.get("providers", []):
        if provider.get("id") == key_id:
            return provider.get("key", "")
    return ""


def extract_json(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


class LLMJudgeClient:
    """Drop-in replacement for JevClient.judge()."""

    def __init__(self, cfg: Config, provider=None):
        self.cfg = cfg
        self.provider = (
            provider or os.environ.get("CRY_AGENT_JUDGE_PROVIDER", "deepseek")
        ).strip().lower()
        if self.provider not in PROVIDERS:
            raise ValueError(
                f"unknown judge provider {self.provider!r}; "
                f"choose from {sorted(PROVIDERS)}"
            )
        self.tier_name = self.provider
        preset = PROVIDERS[self.provider]
        self.base_url = os.environ.get(
            "CRY_AGENT_JUDGE_BASE_URL", preset["base_url"]
        ).rstrip("/")
        self.model = os.environ.get(
            "CRY_AGENT_JUDGE_MODEL", preset["model"]
        )
        self.api_key = (
            os.environ.get("CRY_AGENT_JUDGE_API_KEY")
            or load_provider_key(preset.get("key_id"))
        )
        self.json_mode = preset.get("json_mode", False)

    def _prompt(self, question, choices, state):
        criteria = {
            str(c): CAUSE_CRITERIA.get(str(c))
            or f"The evidence indicates {str(c).replace('_', ' ')}."
            for c in choices
        }
        evidence = json.dumps(state or {}, ensure_ascii=False, default=str)
        if len(evidence) > 4000:
            evidence = evidence[:4000] + "..."
        system = (
            "You are a decision model for a baby-cry monitor. Use only the "
            "evidence. Choose exactly one allowed option. Reply with JSON "
            "only, no prose: "
            '{"choice": "<option>", "confidence": <0.0-1.0>, '
            '"reason": "<one short sentence>"}'
        )
        user = (
            f"Observation:\n{question}\n\n"
            f"Structured evidence:\n{evidence}\n\n"
            "Allowed options:\n"
            + "\n".join(f"- {c}: {d}" for c, d in criteria.items())
        )
        return {"system": system, "user": user}

    def _post(self, prompt, json_mode):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            "temperature": 0,
            "max_tokens": 160,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        timeout = 60 if "127.0.0.1" in self.base_url else 25
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def judge(self, question: str, choices: list[str], state: dict) -> dict:
        t0 = time.perf_counter()
        if not choices:
            return self._error("choices must not be empty", t0)
        prompt = self._prompt(question, choices, state)

        data = None
        error = None
        for attempt, json_mode in enumerate(
            [self.json_mode, False] if self.json_mode else [False]
        ):
            try:
                data = self._post(prompt, json_mode)
                break
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    body = json.loads(e.read())
                    detail = str(body.get("error") or body.get("detail") or "")
                except Exception:
                    pass
                error = f"http_{e.code} {detail}".strip()
                if e.code == 400 and json_mode:
                    continue  # provider rejects response_format
                break
            except Exception as e:
                error = str(e)
                if attempt == 0:
                    time.sleep(0.4)
                    continue
                break

        if data is None:
            return self._error(error or "request failed", t0)

        try:
            content = data["choices"][0]["message"]["content"]
            parsed = extract_json(content)
        except Exception as e:
            return self._error(f"bad_json: {e}", t0)

        raw_choice = parsed.get("choice") or parsed.get("answer")
        choice = self._match_choice(raw_choice, choices)
        if choice is None:
            return self._error(f"invalid_choice: {raw_choice!r}", t0)

        try:
            confidence = float(parsed.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = min(0.99, max(0.01, confidence))

        rest = (1.0 - confidence) / max(len(choices) - 1, 1)
        probabilities = {
            str(c): (confidence if c == choice else rest) for c in choices
        }
        ranked = sorted(probabilities.items(), key=lambda kv: -kv[1])
        return {
            "answer": choice,
            "confidence": round(confidence, 3),
            "alternatives": [
                {"answer": c, "confidence": round(p, 3)}
                for c, p in ranked if c != choice
            ][:3],
            "probabilities": {c: round(p, 3) for c, p in probabilities.items()},
            "reason": parsed.get("reason", ""),
            "model": data.get("model", self.model),
            "usage": data.get("usage", {}),
            "provider": self.provider,
            "route": self.base_url,
            "calibrated": False,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    @staticmethod
    def _match_choice(value, choices):
        if value is None:
            return None
        text = str(value).strip().lower()
        for choice in choices:
            if text == str(choice).lower():
                return choice
        return None

    def _error(self, message, t0):
        return {
            "answer": None,
            "confidence": 0.0,
            "error": message,
            "model": f"{self.provider}-error",
            "provider": self.provider,
            "route": self.base_url,
            "calibrated": False,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }


def _smoke(provider):
    from config import load_config

    client = LLMJudgeClient(load_config(), provider)
    result = client.judge(
        "Baby cried for 20 minutes. Fed 3.2 hours ago, slept 2 hours ago. "
        "Audio classifier: hungry 0.62, discomfort 0.21. What is the cause?",
        ["hungry", "sleepy", "discomfort", "pain", "burp"],
        {
            "audio_cause": "hungry",
            "audio_confidence": 0.62,
            "context": {"hours_since_feed": 3.2, "hours_since_sleep": 2.0},
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import sys

    _smoke(sys.argv[1] if len(sys.argv) > 1 else None)
