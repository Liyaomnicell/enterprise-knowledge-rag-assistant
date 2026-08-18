import re


class SimpleQueryRewriter:

    STOP_WORDS = {
        "what",
        "which",
        "how",
        "should",
        "is",
        "are",
        "the",
        "a",
        "an",
        "to",
        "for",
        "after",
        "happen",
        "occasionally",
        "expected",
    }

    def rewrite(
        self,
        query: str,
    ) -> str:

        tokens = re.findall(
            r"[a-z0-9-]+",
            query.lower(),
        )

        filtered_tokens = [
            token
            for token in tokens
            if token not in self.STOP_WORDS
        ]

        return " ".join(
            filtered_tokens
        )