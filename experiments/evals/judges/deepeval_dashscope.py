from __future__ import annotations

import json
from typing import Any

from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel, ValidationError

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
        if schema is None:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content or ""

        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        structured_prompt = (
            f"{prompt}\n\n"
            "Return ONLY a valid JSON object matching this JSON schema. "
            "Do not include markdown fences or extra text.\n"
            f"JSON schema: {schema_json}"
        )
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": structured_prompt}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "extra_body": {"enable_thinking": False},
        }

        response = self.client.chat.completions.create(**request)
        content = response.choices[0].message.content or ""

        try:
            return schema.model_validate_json(content)
        except ValidationError as error:
            retry_prompt = (
                f"{structured_prompt}\n\n"
                "The previous response failed schema validation. "
                "Return a corrected JSON object only.\n"
                f"Validation error: {error}"
            )
            retry_request = {
                **request,
                "messages": [{"role": "user", "content": retry_prompt}],
            }
            retry_response = self.client.chat.completions.create(**retry_request)
            retry_content = retry_response.choices[0].message.content or ""
            return schema.model_validate_json(retry_content)

    async def a_generate(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        return self.generate(prompt, schema=schema)
