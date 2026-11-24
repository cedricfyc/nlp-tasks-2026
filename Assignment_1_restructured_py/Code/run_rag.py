"""Ask questions from the terminal.

    python run_rag.py                                  # the team's 4 test questions
    python run_rag.py "What is a blocked account?"     # your own question
    python run_rag.py --splade --reranker              # with SPLADE and the re-ranker
"""

import argparse

from pipeline import RAGPipeline

TEST_QUESTIONS = [
    "What legal requirements apply to Eigenbedarf?",
    "How long does a Mietkaution take to be received back? What should I do if I have still not received it?",
    "What are the requirements for a blocked account in Germany? Why is it necessary?",
    "I am a gay person in a wheelchair. How can I protect myself against discrimination in Germany?",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", nargs="*", default=TEST_QUESTIONS)
    parser.add_argument("--splade", action="store_true", help="add SPLADE to the search")
    parser.add_argument("--reranker", action="store_true", help="re-rank with Qwen3-Reranker")
    args = parser.parse_args()

    rag = RAGPipeline(use_splade=args.splade, use_reranker=args.reranker)
    for question in args.questions:
        print("\n" + "=" * 80 + f"\nQ: {question}")
        rag.conduct_rag(question)
