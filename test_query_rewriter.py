from app.rag.query_rewriter import (
    SimpleQueryRewriter,
)


def main():

    rewriter = SimpleQueryRewriter()

    test_queries = [
        {
            "id": "api_003",
            "query": (
                "What processing approach is "
                "recommended for long-running "
                "API work?"
            ),
        },
        {
            "id": "api_004",
            "query": (
                "Requests are occasionally taking "
                "much longer than expected. "
                "What areas should engineers inspect "
                "to identify the bottleneck?"
            ),
        },
        {
            "id": "retry_004",
            "query": (
                "What should happen after a "
                "downstream service returns HTTP 503?"
            ),
        },
    ]

    print()
    print("=" * 80)
    print("QUERY REWRITER TEST")
    print("=" * 80)

    for item in test_queries:

        original_query = item["query"]

        rewritten_query = rewriter.rewrite(
            original_query
        )

        print()
        print("-" * 80)

        print(
            f"ID: {item['id']}"
        )

        print()

        print("Original:")
        print(original_query)

        print()

        print("Rewritten:")
        print(rewritten_query)


if __name__ == "__main__":
    main()
    