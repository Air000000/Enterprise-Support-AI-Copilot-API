from __future__ import annotations

import json
from typing import Any

from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel

from rag_runtime.query_rag_chroma import get_llm_client


class DashScopeDeepEvalModel(DeepEvalBaseLLM):
    """Thin DeepEval adapter for the project's DashScope OpenAI-compatible client."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        model_name: str = "qwen3.5-plus",
    ) -> None:
        self.client = client or get_llm_client()
        self.model_name = model_name

    def load_model(self) -> Any:
        return self.client

    def get_model_name(self) -> str:
        return self.model_name

    def generate(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        messages = [{"role": "user", "content": prompt}]
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
        }

        if schema is not None:
            schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
            messages[0]["content"] = (
                f"{prompt}\n\n"
                "Return ONLY a valid JSON object matching this JSON schema. "
                "Do not include markdown fences or extra text.\n"
                f"JSON schema: {schema_json}"
            )
            request["response_format"] = {"type": "json_object"}
            request["extra_body"] = {"enable_thinking": False}

        response = self.client.chat.completions.create(**request)
        content = response.choices[0].message.content or ""

        if schema is None:
            return content

        return schema.model_validate_json(content)

    async def a_generate(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        return self.generate(prompt, schema=schema)
