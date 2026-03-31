from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.evals.judge_prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt

settings = get_settings()


class JudgeResult(BaseModel):
    relevance_score: int = Field(ge=1, le=5)
    groundedness_score: int = Field(ge=1, le=5)
    completeness_score: int = Field(ge=1, le=5)
    passed: bool
    reasoning: str


class LLMJudge:
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for eval judging")

        self.model = ChatOpenAI(
            model=settings.openai_reasoning_model,
            api_key=settings.openai_api_key,
            temperature=0,
            use_responses_api=True,
        ).with_structured_output(JudgeResult)

    async def judge(
        self,
        *,
        task_type: str,
        query: str,
        expected_behavior: list[str],
        output_payload: dict,
    ) -> JudgeResult:
        prompt = build_judge_user_prompt(
            task_type=task_type,
            query=query,
            expected_behavior=expected_behavior,
            output_payload=output_payload,
        )
        return await self.model.ainvoke(
            [
                ("system", JUDGE_SYSTEM_PROMPT),
                ("human", prompt),
            ]
        )