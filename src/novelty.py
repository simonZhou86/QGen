import re
from typing import Any

from .llm_client import OpenAICompatibleClient


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def lexical_similarity(a: str, b: str) -> float:
    ta = tokenize(a)
    tb = tokenize(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta.intersection(tb))
    union = len(ta.union(tb))
    return inter / union


def is_novel(
    candidate_question: str,
    history: list[dict[str, Any]],
    client: OpenAICompatibleClient,
    lexical_threshold: float = 0.8,
    llm_window: int = 15,
) -> tuple[bool, dict[str, Any]]:
    if not history:
        return True, {"method": "cold_start"}

    scored: list[tuple[float, str]] = []
    for row in history:
        old_question = row.get("question", "")
        if not old_question:
            continue
        sim = lexical_similarity(candidate_question, old_question)
        scored.append((sim, old_question))

    if not scored:
        return True, {"method": "cold_start_after_filter"}

    scored.sort(key=lambda x: x[0], reverse=True)
    best_sim, best_match = scored[0]
    if best_sim >= lexical_threshold:
        return False, {
            "method": "lexical",
            "max_similarity": round(best_sim, 4),
            "closest_question": best_match,
        }

    top_refs = [item[1] for item in scored[:llm_window]]
    verdict = client.chat_json(
        system_prompt=(
            "You are a strict novelty checker to compare new generated question with the history questions. "
            "Return JSON with keys: is_novel (bool), confidence (0-1), rationale (string)."
            "You have to follow the return format, so that the output can be loaded directly using json.load() in Python"
        ),
        user_prompt=(
            "Candidate question:\n"
            f"{candidate_question}\n\n"
            "Previous questions:\n"
            + "\n".join([f"- {ref}" for ref in top_refs])
            + "\n\nIf candidate is materially the same as any previous question, set is_novel=false."
        ),
        temperature=0.0,
    )

    return bool(verdict.get("is_novel", False)), {
        "method": "llm",
        "max_similarity": round(best_sim, 4),
        "verdict": verdict,
    }
