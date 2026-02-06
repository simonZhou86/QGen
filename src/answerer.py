from .llm_client import OpenAICompatibleClient


class QuestionAnswerer:
    def __init__(self, client: OpenAICompatibleClient) -> None:
        self.client = client

    def answer(self, question: str, contexts: list[dict[str, str]], task_type: str = "") -> str:
        context_text = "\n\n".join(
            [
                (
                    f"[source={ctx['source_id']} chunk={ctx['chunk_id']} title={ctx['title']}]\n"
                    f"{ctx['text']}"
                )
                for ctx in contexts
            ]
        )
        output_rule = (
            "If this is a claim quality check task, format answer as "
            "'Verdict: FAITHFUL/HALLUCINATED/NOT_MENTIONED' then one short evidence rationale."
            if task_type == "claim_quality_check"
            else "Answer the question clearly and concisely based on the context you received, you should not make up any content."
        )
        return self.client.chat_text(
            system_prompt=(
                "You are a life sciences practitioner who has a wide range of knowledge in the field of life sciences and medicine. "
                "Answer using only the provided context. If evidence is missing, say insufficient evidence. "
                f"{output_rule}"
            ),
            user_prompt=f"Question:\n{question}\n\nContext:\n{context_text}",
            temperature=0,
        )
