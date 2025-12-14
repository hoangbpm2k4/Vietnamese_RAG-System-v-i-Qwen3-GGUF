"""
Example Usage của Vietnamese RAG System

Demonstrates:
- Basic query
- Batch queries
- Custom configuration
- Error handling
"""

import os
from rag_ultimate_v2 import answer, build_index_if_needed, init_model

def example_basic_query():
    """Example 1: Basic query"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Query")
    print("=" * 60)

    result = answer("Các bước vận hành máy phay CNC là gì?")

    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nRetrieved {len(result['retrieved_chunks'])} chunks")
    print(f"Top score: {result['retrieved_scores'][0]:.3f}")


def example_batch_queries():
    """Example 2: Batch multiple queries"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Batch Queries")
    print("=" * 60)

    questions = [
        "Bước offset dao là gì?",
        "Tại sao cần kiểm tra dầu tưới nguội?",
        "Các bước chuẩn bị máy phay CNC?",
    ]

    # Pre-load model
    init_model()

    for i, q in enumerate(questions, 1):
        print(f"\n--- Query {i} ---")
        print(f"Q: {q}")

        result = answer(q)
        print(f"A: {result['answer'][:200]}...")


def example_custom_config():
    """Example 3: Custom configuration via environment variables"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Custom Configuration")
    print("=" * 60)

    # Set custom params
    os.environ["K_TOP"] = "5"  # Retrieve top-5 instead of 3
    os.environ["TEMPERATURE"] = "0.1"  # More deterministic

    # Note: Need to reload module for env changes to take effect
    print("\nConfiguration:")
    print(f"  K_TOP: {os.getenv('K_TOP')}")
    print(f"  TEMPERATURE: {os.getenv('TEMPERATURE')}")

    result = answer("Quy trình gia công phay CNC gồm mấy bước?")
    print(f"\nAnswer: {result['answer']}")


def example_error_handling():
    """Example 4: Error handling"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Error Handling")
    print("=" * 60)

    try:
        # Query không có trong tài liệu
        result = answer("Cách làm bánh pizza?")
        print(f"\nAnswer: {result['answer']}")

    except Exception as e:
        print(f"Error: {e}")


def main():
    # Build index if needed
    print("Checking index...")
    build_index_if_needed()

    # Run examples
    example_basic_query()

    # Uncomment để chạy thêm examples
    # example_batch_queries()
    # example_custom_config()
    # example_error_handling()


if __name__ == "__main__":
    main()
