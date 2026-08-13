import os
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI

from app.rag.models import RetrievalResult


load_dotenv()


class EvidenceSufficiency(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"


class EvidenceSufficiencyChecker:
    """
    Determine whether retrieved evidence contains
    enough explicit information to answer a question.

    This component does NOT generate the final answer.
    It only classifies evidence as:

        SUFFICIENT
        INSUFFICIENT
    """

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model = os.getenv("OPENAI_MODEL")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        if not model:
            raise ValueError(
                "OPENAI_MODEL is not configured."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self.model = model

    def check(
        self,
        question: str,
        retrieval_results: list[RetrievalResult],
    ) -> EvidenceSufficiency:

        if not retrieval_results:
            return EvidenceSufficiency.INSUFFICIENT

        context = self._build_context(
            retrieval_results
        )

        instructions = """
You are an evidence sufficiency classifier for an enterprise RAG system.

Your task is NOT to answer the user's question.

Your task is to determine whether the provided context contains enough explicit information to answer the question reliably.

Rules:

1. Use only the provided context.
2. Do not use external knowledge.
3. Do not infer missing specific facts.
4. If the question asks for an exact value, number, product, duration, policy, configuration, or other specific fact, the context must explicitly provide that information.
5. Related information is not necessarily sufficient information.
6. If the context only discusses the topic generally but does not contain the requested answer, classify it as INSUFFICIENT.
7. If the answer can be directly supported by the provided context, classify it as SUFFICIENT.

Return exactly one word:

SUFFICIENT

or

INSUFFICIENT
"""

        prompt = f"""
QUESTION:

{question}

CONTEXT:

{context}
"""

        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt,
        )

        output = response.output_text.strip().upper()

        return self._parse_result(output)

    def _build_context(
        self,
        retrieval_results: list[RetrievalResult],
    ) -> str:

        context_parts = []

        for index, result in enumerate(
            retrieval_results,
            start=1,
        ):
            context_parts.append(
                f"""
SOURCE {index}

Document:
{result.chunk.document_name}

Chunk ID:
{result.chunk.chunk_id}

Similarity Score:
{result.score:.4f}

Content:
{result.chunk.content}
"""
            )

        return "\n".join(context_parts)

    def _parse_result(
        self,
        output: str,
    ) -> EvidenceSufficiency:

        # Check INSUFFICIENT first because
        # "INSUFFICIENT" contains the substring
        # "SUFFICIENT".
        if "INSUFFICIENT" in output:
            return EvidenceSufficiency.INSUFFICIENT

        if "SUFFICIENT" in output:
            return EvidenceSufficiency.SUFFICIENT

        raise ValueError(
            "Unexpected evidence sufficiency "
            f"classifier output: {output}"
        )