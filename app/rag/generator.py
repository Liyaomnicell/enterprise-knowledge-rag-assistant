import os

from dotenv import load_dotenv
from openai import OpenAI

from app.rag.models import RetrievalResult


load_dotenv()


class AnswerGenerator:

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def generate(
        self,
        question: str,
        retrieval_results: list[RetrievalResult],
    ) -> str:

        context = self._build_context(
            retrieval_results
        )

        instructions = """
You are an enterprise engineering knowledge assistant.

Answer the user's question using only the provided context.

Rules:
1. Do not use external knowledge.
2. If the context does not contain enough information, say:
   "I don't have enough information in the knowledge base to answer this question."
3. Do not invent facts.
4. Keep the answer concise and technical.
5. Base the answer on the retrieved engineering documents.
"""

        prompt = f"""
QUESTION:

{question}

CONTEXT:

{context}
"""

        response = self.client.responses.create(
            model="gpt-5.5",
            instructions=instructions,
            input=prompt,
        )

        return response.output_text

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
Document: {result.chunk.document_name}
Chunk ID: {result.chunk.chunk_id}

{result.chunk.content}
"""
            )

        return "\n".join(context_parts)
