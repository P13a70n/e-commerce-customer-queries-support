import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from langchain_google_genai import ChatGoogleGenerativeAI

from app import (
    build_llm,
    explain_provider_error,
    get_theme_css,
    load_knowledge,
    retrieve_knowledge,
    split_category,
)


class SupportAppTests(unittest.TestCase):
    def test_csv_knowledge_has_policies_intents_and_orders(self):
        knowledge = load_knowledge()
        types = {record["type"] for record in knowledge}
        self.assertIn("policy", types)
        self.assertIn("intent", types)
        self.assertIn("sample_order", types)

    def test_csv_knowledge_rejects_incomplete_rows(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge.csv"
            path.write_text("type,title,content\npolicy,Returns,\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "row 2"):
                load_knowledge(path)

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

    def test_build_llm_configures_gemini(self):
        with patch.dict(
            "os.environ",
            {"GEMINI_API_KEY": "test-key"},
            clear=True,
        ):
            llm = build_llm()
        self.assertIsInstance(llm, ChatGoogleGenerativeAI)
        self.assertEqual(llm.model, "gemini-3.1-flash-lite")

    def test_gemini_client_has_a_bounded_request_timeout_and_no_retries(self):
        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True),
            patch("app.ChatGoogleGenerativeAI") as gemini_client,
        ):
            build_llm()
        self.assertEqual(
            gemini_client.call_args.kwargs["timeout"],
            20,
        )
        self.assertEqual(gemini_client.call_args.kwargs["max_retries"], 1)

    def test_gemini_requires_an_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
                build_llm()

    def test_gemini_key_can_be_loaded_from_streamlit_deployment_secrets(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "app.st",
                SimpleNamespace(secrets={"GEMINI_API_KEY": "deployment-test-key"}),
            ),
        ):
            llm = build_llm()
        self.assertIsInstance(llm, ChatGoogleGenerativeAI)

    def test_timeout_error_is_actionable(self):
        error = explain_provider_error(TimeoutError("request timed out"))
        self.assertIn("20 seconds", error)

    def test_theme_css_inherits_streamlit_theme_colors(self):
        css = get_theme_css()
        self.assertIn("color: inherit", css)
        self.assertIn("currentColor", css)
        self.assertIn('[data-testid="stChatInput"] textarea', css)
        self.assertIn("button svg", css)
        self.assertIn('[data-testid="stSidebarCollapsedControl"] svg', css)
        self.assertNotIn("fill: currentColor", css)
        self.assertNotIn("stroke: currentColor", css)
        self.assertNotIn("var(--text-color)", css)
        self.assertNotIn("var(--background-color)", css)
        self.assertNotIn("var(--secondary-background-color)", css)


if __name__ == "__main__":
    unittest.main()