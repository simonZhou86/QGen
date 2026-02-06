from typing import Any

from .llm_client import OpenAICompatibleClient


class QuestionGenerator:
    def __init__(self, client: OpenAICompatibleClient) -> None:
        self.client = client

    def generate(
        self,
        target: dict[str, str],
        recent_questions: list[str],
        contexts: list[dict[str, str]],
        temperature: float | None = None,
    ) -> dict[str, Any]:
        recent_text = "\n".join([f"- {q}" for q in recent_questions[-10:]]) or "- None"
        context_text = "\n\n".join(
            [
                (
                    f"[source={ctx['source_id']} chunk={ctx['chunk_id']} title={ctx['title']}]\n"
                    f"{ctx['text']}"
                )
                for ctx in contexts
            ]
        )
        result = self.client.chat_json(
            system_prompt=(
                "You are an intelligent question generation agent."
                "You can generate difficult and high-utility benchmark questions grounded in provided source excerpts. "
                "Your task is to generate different questions with different difficulty level."
                "Always return valid JSON with keys: "
                "question, gold_answer, reasoning_type, difficulty, task_type, source_id, "
                "source_quote, utility_rationale, quality_risk, claim_text, expected_verdict."
                "Your output is expected to be directly loaded with json.load() in Python"
            ),
            user_prompt=(
                "Create one novel benchmark question.\n"
                f"Target reasoning_type: {target['reasoning_type']}\n"
                f"Target difficulty: {target['difficulty']}\n\n"
                f"Target task_type: {target['task_type']}\n\n"
                "Constraints:\n"
                "1) Ground question and answer in Sources below.\n"
                "2) Include a clear gold answer for scoring.\n"
                "3) Must not repeat any item in Recent questions.\n"
                "4) Favor questions that reveal model weaknesses.\n"
                "5) source_quote should be a short exact quote from Sources.\n"
                "6) quality_risk should be one of: none, ambiguous, weak_gold, unsupported.\n\n"
                "Task types:\n"
                "- extractive_qa: answer is an explicit span or fact directly from source.\n"
                "- abstractive_qa: answer requires concise synthesis from multiple source facts.\n"
                "- claim_quality_check: create a concrete claim and ask if it is FAITHFUL or HALLUCINATED relative to source.\n\n"
                "Difficulty policy (must strictly follow):\n"
                "For extractive_qa:\n"
                "- difficulty 1: single-sentence fact; answer appears verbatim in source.\n"
                "- difficulty 3: requires resolving reference/condition within the same paragraph.\n"
                "- difficulty 5: requires cross-sentence understanding within one document.\n"
                "For abstractive_qa:\n"
                "- difficulty 1: summarize 2-3 obvious points.\n"
                "- difficulty 3: summarize 4-5 points dispersed across the text.\n"
                "- difficulty 5: infer implicit motivation or design rationale.\n"
                "For claim_quality_check:\n"
                "- difficulty 1: claim is clearly supported or clearly refuted.\n"
                "- difficulty 3: verdict requires understanding experiment conditions or subgroup constraints.\n"
                "- difficulty 5: use NOT_MENTIONED-style uncertainty or subtle contradiction.\n\n"
                "For claim_quality_check:\n"
                "- Put the claim text in claim_text.\n"
                "- expected_verdict must be exactly one of: FAITHFUL, HALLUCINATED, NOT_MENTIONED.\n"
                "- gold_answer should start with 'Verdict: ...' then short evidence rationale.\n"
                "For non-claim tasks, set claim_text to empty string and expected_verdict to NA.\n\n"
                "Recent questions:\n"
                f"{recent_text}\n\n"
                "Sources:\n"
                f"{context_text}"
            ),
            temperature=0.7 if temperature is None else temperature,
        )
        return result
