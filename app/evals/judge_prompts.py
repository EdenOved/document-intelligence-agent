from __future__ import annotations


JUDGE_SYSTEM_PROMPT = """
You are an evaluation judge for an AI document intelligence system.

You must evaluate the quality of a system output for the requested task.

Scoring rules:
- relevance_score: how well the output answers the user's request
- groundedness_score: how well the output is supported by the provided evidence/citations
- completeness_score: how fully the output covers the important points for the task
- passed: true only if the output is strong enough overall

Use these score ranges:
1 = very poor
2 = weak
3 = partially acceptable
4 = strong
5 = excellent

Important rules:
- Be strict.
- Penalize hallucinations.
- Penalize weak or missing evidence.
- Penalize shallow compare outputs.
- Penalize missing key task coverage.
- If the output is unsupported by citations/evidence, groundedness must be low.
- Return only structured JSON matching the schema.
""".strip()


def build_judge_user_prompt(
    *,
    task_type: str,
    query: str,
    expected_behavior: list[str],
    output_payload: dict,
) -> str:
    return f"""
Task type:
{task_type}

User query:
{query}

Expected behavior:
{expected_behavior}

System output:
{output_payload}

Evaluate the output for relevance, groundedness, and completeness.
""".strip()