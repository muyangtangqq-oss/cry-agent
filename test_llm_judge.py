"""Tests for the OpenAI-compatible cloud judge (no network)."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config import load_config
from judge_factory import build_cloud_judge
from llm_judge import LLMJudgeClient, extract_json


class StubHandler(BaseHTTPRequestHandler):
    response_text = (
        '{"choice":"hungry","confidence":0.8,'
        '"reason":"feed gap and audio agree"}'
    )
    status = 200

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = json.dumps({
            "model": "stub-model",
            "choices": [{
                "message": {"role": "assistant",
                            "content": self.response_text},
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }).encode()
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def with_stub(fn):
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    old = {
        key: os.environ.get(key)
        for key in (
            "CRY_AGENT_JUDGE_PROVIDER",
            "CRY_AGENT_JUDGE_BASE_URL",
            "CRY_AGENT_JUDGE_MODEL",
            "CRY_AGENT_JUDGE_API_KEY",
        )
    }
    os.environ.update({
        "CRY_AGENT_JUDGE_PROVIDER": "ollama",
        "CRY_AGENT_JUDGE_BASE_URL": f"http://127.0.0.1:{port}/v1",
        "CRY_AGENT_JUDGE_MODEL": "stub-model",
        "CRY_AGENT_JUDGE_API_KEY": "",
    })
    try:
        return fn()
    finally:
        server.shutdown()
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_judge():
    def run():
        client = LLMJudgeClient(load_config(), "ollama")
        result = client.judge(
            "Audio: hungry 62%, fed 3.2h ago.",
            ["hungry", "sleepy", "discomfort", "pain", "burp"],
            {"audio_cause": "hungry", "audio_confidence": 0.62},
        )
        assert result["answer"] == "hungry", result
        assert abs(result["confidence"] - 0.8) < 1e-6, result
        assert abs(sum(result["probabilities"].values()) - 1.0) < 0.01
        assert result["model"] == "stub-model"
        assert result["provider"] == "ollama"
        assert result["calibrated"] is False
        assert result["reason"]
        return result
    return with_stub(run)


def test_bad_json():
    def run():
        StubHandler.response_text = "not json at all"
        try:
            client = LLMJudgeClient(load_config(), "ollama")
            result = client.judge(
                "Audio: hungry.", ["hungry", "sleepy"], {}
            )
            assert result["answer"] is None, result
            assert "bad_json" in result["error"], result
        finally:
            StubHandler.response_text = (
                '{"choice":"hungry","confidence":0.8,'
                '"reason":"feed gap and audio agree"}'
            )
    return with_stub(run)


def test_extract_json():
    assert extract_json("```json\n{\"choice\":\"hungry\"}\n```") == {
        "choice": "hungry"
    }
    assert extract_json('prefix {"choice":"pain"} suffix') == {
        "choice": "pain"
    }


def test_factory():
    old = os.environ.get("CRY_AGENT_JUDGE_PROVIDER")
    try:
        os.environ["CRY_AGENT_JUDGE_PROVIDER"] = "jev"
        assert build_cloud_judge(load_config()).tier_name == "jev"
        os.environ["CRY_AGENT_JUDGE_PROVIDER"] = "deepseek"
        judge = build_cloud_judge(load_config())
        assert judge.tier_name == "deepseek"
        assert judge.model
    finally:
        if old is None:
            os.environ.pop("CRY_AGENT_JUDGE_PROVIDER", None)
        else:
            os.environ["CRY_AGENT_JUDGE_PROVIDER"] = old


if __name__ == "__main__":
    r = test_judge()
    print(f"stub judge: {r['answer']} {r['confidence']} "
          f"model={r['model']}")
    test_bad_json()
    print("bad JSON -> error path OK")
    test_extract_json()
    print("extract_json OK")
    test_factory()
    print("factory OK")
    print("LLM judge tests passed.")
