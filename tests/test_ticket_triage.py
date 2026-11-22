import unittest
from unittest.mock import patch

from src.ticket_triage import (
    redact_pii,
    count_pii,
    call_model_with_retry,
)


class TestTicketPIIRedaction(unittest.TestCase):

    def test_email_is_redacted(self):
        result = redact_pii("Contact user@example.com for help")
        self.assertEqual(result, "Contact [EMAIL] for help")

    def test_phone_is_redacted(self):
        result = redact_pii("Call 9876543210 for support")
        self.assertEqual(result, "Call [PHONE] for support")

    def test_order_id_is_redacted(self):
        result = redact_pii("Order ORD-123456 is delayed")
        self.assertEqual(result, "Order [ORDER_ID] is delayed")

    def test_all_supported_pii_is_redacted(self):
        text = "user@example.com 9876543210 ORD-123456"
        result = redact_pii(text)

        self.assertEqual(
            count_pii(result),
            {"emails": 0, "phones": 0, "order_ids": 0},
        )

    def test_pii_counts_before_redaction(self):
        text = "user@example.com 9876543210 ORD-123456"

        self.assertEqual(
            count_pii(text),
            {"emails": 1, "phones": 1, "order_ids": 1},
        )

    def test_empty_text(self):
        self.assertEqual(redact_pii(""), "")

    def test_missing_text(self):
        self.assertEqual(redact_pii(None), "")

class TestTicketPromptPrivacy(unittest.TestCase):

    def test_user_prompt_with_pii_is_rejected(self):
        with patch("src.ticket_triage.mock_llm.chat") as mock_chat:
            with self.assertRaisesRegex(
                AssertionError, "PII detected in user prompt"
            ):
                call_model_with_retry(
                    system_prompt="Classify the ticket",
                    user_text="Contact user@example.com",
                    ticket_id="T0001",
                )

            mock_chat.assert_not_called()

    def test_system_prompt_with_pii_is_rejected(self):
        with patch("src.ticket_triage.mock_llm.chat") as mock_chat:
            with self.assertRaisesRegex(
                AssertionError, "PII detected in system prompt"
            ):
                call_model_with_retry(
                    system_prompt="Contact user@example.com",
                    user_text="My order is delayed",
                    ticket_id="T0002",
                )

            mock_chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
