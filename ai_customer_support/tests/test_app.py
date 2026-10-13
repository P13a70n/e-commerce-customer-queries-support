import unittest
from unittest.mock import patch

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app import (
    build_llm,
    explain_provider_error,
    load_knowledge,
    retrieve_knowledge,
    split_category,
)


class SupportAppTests(unittest.TestCase):
    def test_demo_knowledge_has_policies_intents_and_orders(self):
        knowledge = load_knowledge()
        self.assertTrue(knowledge["policies"])
        self.assertTrue(knowledge["intents"])
        self.assertTrue(knowledge["sample_orders"])

    def test_retrieval_returns_the_matching_sample_order(self):
        retrieved = retrieve_knowledge(
            "Where is my order NS-1048?", load_knowledge()
        )
        self.assertIn("NS-1048", retrieved)
        self.assertIn("In transit; delayed by severe weather", retrieved)
        self.assertNotIn("NS-2091", retrieved)

    def test_retrieval_returns_the_relevant_policy(self):
        retrieved = retrieve_knowledge(
            "How long do I have to return an unused item?", load_knowledge()
        )
        self.assertIn("Policy: Returns", retrieved)
        self.assertIn("within 30 days of delivery", retrieved)

    def test_retrieval_returns_no_documents_for_unmatched_queries(self):
        retrieved = retrieve_knowledge(
            "Tell me a joke about the moon", load_knowledge()
        )
        self.assertEqual(retrieved, "")

    def test_response_category_is_separated_from_answer(self):
        category, answer = split_category(
            "Category: order_status\nYour sample order is in transit."
        )
        self.assertEqual(category, "order_status")
        self.assertEqual(answer, "Your sample order is in transit.")

    def test_unstructured_response_is_preserved(self):
        response = "Please share your order number."
        self.assertEqual(split_category(response), (None, response))

    def test_provider_errors_are_actionable_without_echoing_secrets(self):
        error = explain_provider_error(RuntimeError("401 unauthorized"))
        self.assertIn("credentials", error)
        self.assertNotIn("unauthorized", error)

    def test_build_llm_configures_ollama(self):
        with patch.dict(
            "os.environ",
            {"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": "test-model"},
            clear=True,
        ):
            llm = build_llm()
        self.assertIsInstance(llm, ChatOllama)

    def test_build_llm_configures_openrouter(self):
        with patch.dict(
            "os.environ",
            {
                "LLM_PROVIDER": "openrouter",
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_MODEL": "test/model",
            },
            clear=True,
        ):
            llm = build_llm()
        self.assertIsInstance(llm, ChatOpenAI)

    def test_openrouter_requires_an_api_key(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "openrouter"}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
                build_llm()


if __name__ == "__main__":
    unittest.main()