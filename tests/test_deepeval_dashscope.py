import asyncio
from types import SimpleNamespace

from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel

from experiments.evals.judges.deepeval_dashscope import DashScopeDeepEvalModel


class JudgePayload(BaseModel):
    score: float
    reason: str


class FakeCompletions:
    def __init__(self, contents: list[str]):
        self.contents = iter(contents)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=next(self.contents))
                )
            ]
        )


class FakeClient:
    def __init__(self, contents: list[str]):
        self.completions = FakeCompletions(contents)
        self.chat = SimpleNamespace(completions=self.completions)


def test_dashscope_deepeval_model_supports_plain_text_generation():
    client = FakeClient(["plain judge output"])
    model = DashScopeDeepEvalModel(
        client=client,
        model_name="qwen3.5-plus",
    )

    assert isinstance(model, DeepEvalBaseLLM)
    assert model.load_model() is client
    assert model.get_model_name() == "qwen3.5-plus"
    assert model.generate("Judge this answer") == "plain judge output"

    call = client.completions.calls[0]
    assert call["model"] == "qwen3.5-plus"
    assert call["temperature"] == 0.0
    assert "response_format" not in call
    assert "extra_body" not in call
    assert call["messages"] == [
        {"role": "user", "content": "Judge this answer"}
    ]


def test_dashscope_deepeval_model_validates_schema_json_output():
    client = FakeClient(['{"score": 0.9, "reason": "grounded"}'])
    model = DashScopeDeepEvalModel(
        client=client,
        model_name="qwen3.5-plus",
    )

    result = model.generate("Judge faithfulness", schema=JudgePayload)

    assert result == JudgePayload(score=0.9, reason="grounded")
    call = client.completions.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["extra_body"] == {"enable_thinking": False}
    assert "JSON" in call["messages"][0]["content"]
    assert "score" in call["messages"][0]["content"]
    assert "reason" in call["messages"][0]["content"]


def test_dashscope_deepeval_model_async_path_matches_sync_contract():
    client = FakeClient(['{"score": 1.0, "reason": "correct"}'])
    model = DashScopeDeepEvalModel(
        client=client,
        model_name="qwen3.5-plus",
    )

    result = asyncio.run(
        model.a_generate("Judge correctness", schema=JudgePayload)
    )

    assert result == JudgePayload(score=1.0, reason="correct")
