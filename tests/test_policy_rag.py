import unittest
from unittest.mock import patch
from types import SimpleNamespace

from src.policy_rag import (
    FALLBACK_ANSWER,
    load_policy_documents,
    split_documents,
    build_retriever,
    answer_question,
)


class TestPolicyRAGGrounding(unittest.TestCase):

    def setUp(self):
        self.documents = load_policy_documents()
        self.chunks = split_documents(self.documents)
        self.vectorizer, self.matrix = build_retriever(self.chunks)

    @patch("src.policy_rag.mock_llm.chat")
    def test_rule_answer_has_actual_supporting_source(self, mock_chat):
        mock_chat.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Mock answer")
                )
            ]
        )

        result = answer_question(
            "What is the late delivery refund policy?",
            self.chunks,
            self.vectorizer,
            self.matrix,
        )

        retrieved_sources = {
            item["source"] for item in result["retrieved"]
        }

        self.assertIn("refund_policy.txt", retrieved_sources)
        self.assertIn("refund_policy.txt", result["sources"])

        refund_context = " ".join(
            item["text"]
            for item in result["retrieved"]
            if item["source"] == "refund_policy.txt"
        ).lower()

        self.assertIn("more than 45 minutes", refund_context)
        self.assertIn("within 7 days", refund_context)

    @patch("src.policy_rag.mock_llm.chat")
    def test_sources_match_retrieved_documents(self, mock_chat):
        mock_chat.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Mock answer")
                )
            ]
        )

        result = answer_question(
            "What happens when a payment fails?",
            self.chunks,
            self.vectorizer,
            self.matrix,
        )

        retrieved_sources = sorted({
            item["source"] for item in result["retrieved"]
        })

        self.assertEqual(result["sources"], retrieved_sources)

    def test_missing_supporting_policy_returns_fallback(self):
        payment_chunks = [
            chunk for chunk in self.chunks
            if chunk["source"] == "payment_policy.txt"
        ]

        vectorizer, matrix = build_retriever(payment_chunks)

        result = answer_question(
            "What is the late delivery refund policy?",
            payment_chunks,
            vectorizer,
            matrix,
        )

        self.assertEqual(result["answer"], FALLBACK_ANSWER)
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["method"], "missing_rule_source")


if __name__ == "__main__":
    unittest.main()
