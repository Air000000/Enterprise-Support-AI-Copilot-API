from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from experiments.evals.judges.deepeval_dashscope import DashScopeDeepEvalModel


class Verdict(BaseModel):
    verdict: str
    reason: str | None = None


class Verdicts(BaseModel):
    verdicts: list[Verdict]


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


def test_structured_generation_retries_once_after_schema_validation_error():
    client = FakeClient(
        [
            '{"verdicts":[{"verdict":"yes","reason":"ok"},"bad item"]}',
            '{"verdicts":[{"verdict":"yes","reason":"ok"}]}',
        ]
    )
    model = DashScopeDeepEvalModel(client=client, model_name="qwen3.5-plus")

    result = model.generate("Judge faithfulness", schema=Verdicts)

    assert result == Verdicts(verdicts=[Verdict(verdict="yes", reason="ok")])
    assert len(client.completions.calls) == 2

    first_call, second_call = client.completions.calls
    assert first_call["response_format"] == {"type": "json_object"}
    assert second_call["response_format"] == {"type": "json_object"}
    assert first_call["extra_body"] == {"enable_thinking": False}
    assert second_call["extra_body"] == {"enable_thinking": False}
    assert "failed schema validation" in second_call["messages"][0]["content"]
    assert "Input should be an object" in second_call["messages"][0]["content"]


def test_structured_generation_raises_after_second_schema_validation_error():
    client = FakeClient(
        [
            '{"verdicts":["first bad item"]}',
            '{"verdicts":["second bad item"]}',
        ]
    )
    model = DashScopeDeepEvalModel(client=client, model_name="qwen3.5-plus")

    with pytest.raises(ValidationError):
        model.generate("Judge faithfulness", schema=Verdicts)

    assert len(client.completions.calls) == 2
