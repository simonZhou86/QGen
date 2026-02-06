from typing import Any

from .llm_client import OpenAICompatibleClient


class AnswerEvaluator:
    def __init__(self, client: OpenAICompatibleClient) -> None:
        self.client = client

    def evaluate(
        self,
        question: str,
        gold_answer: str,
        predicted_answer: str,
        contexts: list[dict[str, str]],
        task_type: str = "",
        expected_verdict: str = "NA",
        source_quote: str = "",
    ) -> dict[str, Any]:
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
                "You are a strict judger that evaluate how accurate is the answer given question based on the source content provided to you. "
                "You will receive a task type, question, ground truth answer, candidate answer, candiadte citation, a optional claim verdict if the task type is claim_quality_check, and the original context."
                "Return JSON with keys: score, is_correct, feedback, error_type, quality_flags, ambiguity_note, claim_verdict_match. "
                "score must be in [0,1]."
                "You must return a valid json so that it can be directly loaded by json.loads() in Python."
            ),
            user_prompt=(
                f"Task type:\n{task_type}\n\n"
                f"Question:\n{question}\n\n"
                f"Gold answer:\n{gold_answer}\n\n"
                f"Candidate answer:\n{predicted_answer}\n\n"
                f"Expected claim verdict (if claim_quality_check):\n{expected_verdict}\n\n"
                f"Generator source quote:\n{source_quote}\n\n"
                f"Source context:\n{context_text}\n\n"
                "Evaluate factual correctness and reasoning quality.\n"
                "If task_type is claim_quality_check, prioritize verdict correctness and evidence grounding.\n"
                "Allowed claim verdict labels are FAITHFUL, HALLUCINATED, NOT_MENTIONED.\n"
                "quality_flags is a list of zero or more from: ambiguous_question, weak_gold, unsupported_by_context."
            ),
            temperature=0.0,
        )

        score = float(result.get("score", 0.0))
        result["score"] = min(1.0, max(0.0, score))
        result["is_correct"] = bool(result.get("is_correct", score >= 0.8))
        result["feedback"] = str(result.get("feedback", ""))
        result["error_type"] = str(result.get("error_type", "none"))
        flags = result.get("quality_flags", [])
        if not isinstance(flags, list):
            flags = []
        result["quality_flags"] = [str(flag) for flag in flags]
        result["ambiguity_note"] = str(result.get("ambiguity_note", ""))
        result["claim_verdict_match"] = bool(result.get("claim_verdict_match", False))
        return result
