"""
Regression and unit tests for Token-Aware Context Management in Local AI Assistant.
Covers:
1. New conversation fitting within budget
2. Long historical conversation exceeding budget with older turns trimmed first
3. Large system prompts and tool schemas accounting
4. Oversized historical tool outputs compression
5. Oversized active tool output capping
6. Assistant tool calls with multiple matching tool responses (atomic sequence preservation)
7. Preservation of latest user message
8. Latest turn exceeding budget raising a clear actionable error
9. HTTP 413 recovery with context reduction and single retry (no infinite loops)
10. Ordinary HTTP 429 rate limit backoff handling
11. Confirmation that SQLite history remains unchanged after in-memory trimming
12. Environment variable configuration of token budgets
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import json
import sqlite3
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import openai
from Tools.tools_schema import Tools
from Utils.utilities import (
    estimate_tokens,
    estimate_message_tokens,
    estimate_tools_tokens,
    estimate_request_tokens,
    get_provider_budgets,
    is_oversized_request_error,
)
from Storage.storage import (
    initialise_db,
    add_conversation,
    save_message,
    get_messages,
)
from simpleAiAss import (
    enforce_context_budget,
    execute_agent_turn,
)


class TestContextManagement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initialise_db()

    def setUp(self):
        self.approved_dirs = {PROJECT_ROOT.resolve()}

    # --- 1. New conversation that fits within the budget ---
    def test_new_conversation_fits_budget(self):
        """A new conversation with system prompt and 1 user message fits easily without trimming."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you today?"}
        ]
        budgeted = enforce_context_budget(
            messages,
            tools=Tools,
            max_request_tokens=6000,
            completion_reservation=1500
        )
        self.assertEqual(len(budgeted), 2)
        self.assertEqual(budgeted[0]["role"], "system")
        self.assertEqual(budgeted[1]["role"], "user")
        self.assertEqual(budgeted[1]["content"], "Hello, how are you today?")

    # --- 2. Long historical conversation exceeding budget trims older turns first ---
    def test_long_conversation_exceeds_budget_trims_older_turns(self):
        """Older complete conversation turns are dropped first, keeping recent history."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Turn 1 question " + "A" * 1500},
            {"role": "assistant", "content": "Turn 1 answer " + "B" * 1500},
            {"role": "user", "content": "Turn 2 question " + "C" * 1500},
            {"role": "assistant", "content": "Turn 2 answer " + "D" * 1500},
            {"role": "user", "content": "Turn 3 question " + "E" * 500},
            {"role": "assistant", "content": "Turn 3 answer " + "F" * 500},
            {"role": "user", "content": "Turn 4 latest question: What was my previous question?"}
        ]

        # Use a budget that can fit system, latest user turn, and Turn 3, but not Turn 1 and 2
        budgeted = enforce_context_budget(
            messages,
            tools=None,
            max_request_tokens=1500,
            completion_reservation=200
        )

        roles = [m["role"] for m in budgeted]
        self.assertEqual(budgeted[0]["role"], "system")
        self.assertEqual(budgeted[-1]["role"], "user")
        self.assertIn("Turn 4 latest question", budgeted[-1]["content"])

        # Turn 1 should be dropped first
        contents = " ".join(str(m.get("content")) for m in budgeted)
        self.assertNotIn("Turn 1 question", contents)
        # Turn 3 should be retained as the most recent historical turn
        self.assertIn("Turn 3", contents)

    # --- 3. Large system prompt and tool schemas accounted for ---
    def test_large_system_prompt_and_tool_schemas(self):
        """Tool schemas and system prompt are accurately accounted for in budget estimation."""
        tools_tokens = estimate_tools_tokens(Tools)
        self.assertGreater(tools_tokens, 1000)
        self.assertLess(tools_tokens, 2500)

        large_system = {
            "role": "system",
            "content": "You are a helpful assistant.\n" + "\n".join(f"- Fact {i}: User preference {i}" for i in range(50))
        }
        user_msg = {"role": "user", "content": "What are my preferences?"}
        messages = [large_system, user_msg]

        total_tokens = estimate_request_tokens(messages, tools=Tools, completion_reservation=1500)
        # Verify the calculation is realistic and includes tool schemas
        self.assertGreater(total_tokens, tools_tokens + 1500)

        # Enforce budget - should fit within default 6000 tokens
        budgeted = enforce_context_budget(messages, tools=Tools, max_request_tokens=6000, completion_reservation=1500)
        self.assertEqual(len(budgeted), 2)
        self.assertIn("Fact 49", budgeted[0]["content"])

    # --- 4. Oversized historical tool outputs compression ---
    def test_oversized_historical_tool_outputs_compression(self):
        """Historical tool outputs exceeding aggressive_tool_cap are truncated with a clear notice."""
        large_tool_content = "File content line\n" * 500  # ~9000 chars
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Read file data.txt"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": large_tool_content},
            {"role": "assistant", "content": "I have read data.txt."},
            {"role": "user", "content": "Now summarize it."}
        ]

        budgeted = enforce_context_budget(
            messages,
            tools=Tools,
            max_request_tokens=6000,
            completion_reservation=1500,
            aggressive_tool_cap=400
        )

        tool_msgs = [m for m in budgeted if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        # Content must be truncated and clearly marked
        self.assertIn("Content truncated", tool_msgs[0]["content"])
        self.assertLess(len(tool_msgs[0]["content"]), 600)
        # Tool call ID must be preserved
        self.assertEqual(tool_msgs[0]["tool_call_id"], "call_1")

    # --- 5. Oversized active tool output capping ---
    def test_oversized_active_tool_output_capping(self):
        """Active turn tool output exceeding current_tool_cap is capped with a clear notice."""
        massive_output = "LOG_ROW: status=OK, id=" * 500  # ~12000 chars
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Run the diagnostic test"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_act", "type": "function", "function": {"name": "execute_command", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_act", "content": massive_output}
        ]

        budgeted = enforce_context_budget(
            messages,
            tools=Tools,
            max_request_tokens=6000,
            completion_reservation=1500,
            current_tool_cap=3000
        )

        active_tool = [m for m in budgeted if m.get("tool_call_id") == "call_act"][0]
        self.assertIn("Output truncated", active_tool["content"])
        self.assertLessEqual(len(active_tool["content"]), 3200)
        self.assertEqual(active_tool["tool_call_id"], "call_act")

    # --- 6. Assistant tool calls with multiple matching tool responses (atomic block) ---
    def test_atomic_tool_sequence_preservation(self):
        """An assistant message with multiple tool calls and its responses are kept or dropped together."""
        tool_calls = [
            {"id": "call_a", "type": "function", "function": {"name": "getCurrentTime", "arguments": "{}"}},
            {"id": "call_b", "type": "function", "function": {"name": "list_directory", "arguments": '{"directory_path": "."}'}},
        ]
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Turn 1: What time is it and list directory?"},
            {"role": "assistant", "content": None, "tool_calls": tool_calls},
            {"role": "tool", "tool_call_id": "call_a", "content": "2026-10-25 12:00:00"},
            {"role": "tool", "tool_call_id": "call_b", "content": "[FILE] test.py\n[FILE] README.md"},
            {"role": "assistant", "content": "Here is the time and files."},
            {"role": "user", "content": "Turn 2: Thank you, now let's write code."}
        ]

        # Case A: Sequence fits
        budgeted_fit = enforce_context_budget(messages, tools=None, max_request_tokens=4000, completion_reservation=200)
        has_assistant_tc = any(m.get("tool_calls") for m in budgeted_fit)
        tool_count = sum(1 for m in budgeted_fit if m.get("role") == "tool")
        self.assertTrue(has_assistant_tc)
        self.assertEqual(tool_count, 2)

        # Case B: Tight budget forces dropping Turn 1 tool sequence
        # It must drop the assistant tool_calls AND both tool responses, NEVER splitting them
        budgeted_tight = enforce_context_budget(messages, tools=None, max_request_tokens=100, completion_reservation=50)
        has_assistant_tc_tight = any(m.get("tool_calls") for m in budgeted_tight)
        tool_count_tight = sum(1 for m in budgeted_tight if m.get("role") == "tool")
        self.assertFalse(has_assistant_tc_tight)
        self.assertEqual(tool_count_tight, 0)
        # Latest user message must still be preserved
        self.assertEqual(budgeted_tight[-1]["content"], "Turn 2: Thank you, now let's write code.")

    # --- 7. Preservation of latest user message ---
    def test_preservation_of_latest_user_message(self):
        """Latest user message is always preserved even under very restrictive budgets."""
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Old prompt " + "Z" * 1000},
            {"role": "assistant", "content": "Old response " + "Y" * 1000},
            {"role": "user", "content": "CRITICAL LATEST MESSAGE THAT MUST SURVIVE"}
        ]
        budgeted = enforce_context_budget(messages, tools=None, max_request_tokens=800, completion_reservation=100)
        self.assertEqual(budgeted[-1]["role"], "user")
        self.assertEqual(budgeted[-1]["content"], "CRITICAL LATEST MESSAGE THAT MUST SURVIVE")
        self.assertEqual(budgeted[0]["role"], "system")

    # --- 8. Latest turn cannot fit within budget raises clear actionable error ---
    def test_latest_turn_cannot_fit_raises_clear_error(self):
        """If the latest user prompt plus system prompt exceeds budget, a clear ValueError is raised."""
        huge_user_prompt = "Explain this code in detail: " + "K" * 25000  # ~7800 tokens alone
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": huge_user_prompt}
        ]

        with self.assertRaises(ValueError) as ctx:
            enforce_context_budget(
                messages,
                tools=Tools,
                max_request_tokens=6000,
                completion_reservation=1500
            )

        err_msg = str(ctx.exception)
        self.assertIn("exceed the model request limit", err_msg)
        self.assertIn("Please shorten your input", err_msg)

    # --- 9. HTTP 413 recovery without repeated identical retries ---
    def test_http_413_recovery_without_repeated_identical_retries(self):
        """On HTTP 413, context is reduced to minimal turn and retried ONCE, then fails cleanly if still rejected."""
        mock_client = MagicMock()
        oversized_err = openai.RateLimitError(
            message="Request too large for model `openai/gpt-oss-120b` on tokens per minute (TPM): Limit 8000, Requested 8135",
            response=MagicMock(status_code=413),
            body={"error": {"code": "rate_limit_exceeded", "type": "tokens", "message": "Request too large"}}
        )

        # Sub-case A: 413 on attempt 0, succeeds on attempt 1
        mock_success_response = MagicMock()
        mock_success_response.choices = [
            MagicMock(
                finish_reason="stop",
                message=MagicMock(role="assistant", content="Recovered after context trimming!", tool_calls=None)
            )
        ]
        mock_client.chat.completions.create.side_effect = [oversized_err, mock_success_response]

        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Old turn 1"},
            {"role": "assistant", "content": "Old answer 1"},
            {"role": "user", "content": "Latest user question"}
        ]

        content, updated_messages = execute_agent_turn(
            messages=messages,
            conversation_id=None,
            client_instance=mock_client,
            model_name="openai/gpt-oss-120b",
            approved_dirs=self.approved_dirs,
            epochs=2,
            interactive=False
        )

        self.assertEqual(content, "Recovered after context trimming!")
        # Client was called exactly twice (1 retry, not infinite loop)
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

        # Sub-case B: Persistent 413 on both attempts stops after 1 retry
        mock_client.reset_mock()
        mock_client.chat.completions.create.side_effect = oversized_err

        content_fail, _ = execute_agent_turn(
            messages=messages,
            conversation_id=None,
            client_instance=mock_client,
            model_name="openai/gpt-oss-120b",
            approved_dirs=self.approved_dirs,
            epochs=2,
            interactive=False
        )

        self.assertIn("exceeds model's token throughput limit", content_fail)
        # Should stop after at most 2 calls (attempt 0 and attempt 1 retry)
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    # --- 10. Ordinary HTTP 429 rate limit handling ---
    @patch("time.sleep", return_value=None)
    def test_ordinary_http_429_handling(self, mock_sleep):
        """Ordinary HTTP 429 (requests per minute rate limit) pauses and retries with backoff."""
        mock_client = MagicMock()
        rpm_err = openai.RateLimitError(
            message="Rate limit reached for requests per minute (RPM). Please wait.",
            response=MagicMock(status_code=429),
            body={"error": {"code": "rate_limit_exceeded", "type": "requests"}}
        )

        mock_success_response = MagicMock()
        mock_success_response.choices = [
            MagicMock(
                finish_reason="stop",
                message=MagicMock(role="assistant", content="Responded after rate limit wait.", tool_calls=None)
            )
        ]
        mock_client.chat.completions.create.side_effect = [rpm_err, mock_success_response]

        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Question"}
        ]

        content, _ = execute_agent_turn(
            messages=messages,
            conversation_id=None,
            client_instance=mock_client,
            model_name="openai/gpt-oss-120b",
            approved_dirs=self.approved_dirs,
            epochs=2,
            interactive=False
        )

        self.assertEqual(content, "Responded after rate limit wait.")
        # Verified sleep was called for backoff
        mock_sleep.assert_called_with(2)

    # --- 11. Confirmation that SQLite history remains unchanged after trimming ---
    def test_sqlite_history_remains_unchanged_after_trimming(self):
        """Trimming only affects the in-memory request payload; SQLite message history is untouched."""
        test_title = "SQLite Preservation Test Conversation"
        conv_id = add_conversation(test_title)

        # Store 6 full messages in SQLite
        db_messages = [
            {"role": "user", "content": "Turn 1 question " + "A" * 500},
            {"role": "assistant", "content": "Turn 1 answer " + "B" * 500},
            {"role": "user", "content": "Turn 2 question " + "C" * 500},
            {"role": "assistant", "content": "Turn 2 answer " + "D" * 500},
            {"role": "user", "content": "Turn 3 question " + "E" * 500},
            {"role": "assistant", "content": "Turn 3 answer " + "F" * 500},
        ]
        for m in db_messages:
            save_message(conv_id, m)

        initial_stored = get_messages(conv_id)
        self.assertEqual(len(initial_stored), 6)

        # Now simulate a new turn with execute_agent_turn under a tight budget that trims Turn 1 and 2
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                finish_reason="stop",
                message=MagicMock(role="assistant", content="Turn 4 answer.", tool_calls=None)
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        full_conversation = [{"role": "system", "content": "System prompt"}] + initial_stored + [
            {"role": "user", "content": "Turn 4 question: new query"}
        ]
        save_message(conv_id, {"role": "user", "content": "Turn 4 question: new query"})

        # Execute agent turn with tight 2000 token budget
        content, _ = execute_agent_turn(
            messages=full_conversation,
            conversation_id=conv_id,
            client_instance=mock_client,
            model_name="openai/gpt-oss-120b",
            approved_dirs=self.approved_dirs,
            max_request_tokens=2000,
            completion_reservation=200,
            epochs=2,
            interactive=False
        )

        # Verify SQLite still contains all original messages PLUS the new user & assistant messages
        final_stored = get_messages(conv_id)
        # 6 original + 1 new user + 1 new assistant = 8 total in SQLite
        self.assertEqual(len(final_stored), 8)
        self.assertEqual(final_stored[0]["content"], db_messages[0]["content"])
        self.assertEqual(final_stored[1]["content"], db_messages[1]["content"])
        self.assertEqual(final_stored[-1]["content"], "Turn 4 answer.")

    # --- 12. Environment variable configuration of token budgets ---
    def test_environment_variable_budget_configuration(self):
        """Token budgets default conservatively for Groq and can be overridden via environment variables."""
        groq_info = {"provider": "Groq Cloud", "base_url": "https://api.groq.com/openai/v1"}
        ollama_info = {"provider": "Local Ollama", "base_url": "http://localhost:11434/v1"}

        # Defaults without env overrides
        groq_req, groq_comp = get_provider_budgets(groq_info)
        self.assertEqual(groq_req, 6000)
        self.assertEqual(groq_comp, 1500)

        gemini_info = {"provider": "Google Gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"}
        gemini_req, gemini_comp = get_provider_budgets(gemini_info)
        self.assertEqual(gemini_req, 64000)
        self.assertEqual(gemini_comp, 4096)

        ollama_req, ollama_comp = get_provider_budgets(ollama_info)
        self.assertEqual(ollama_req, 16000)
        self.assertEqual(ollama_comp, 2048)

        # Overrides via env
        with patch.dict(os.environ, {"MAX_REQUEST_TOKENS": "5500", "MAX_COMPLETION_TOKENS": "1200"}):
            ov_req, ov_comp = get_provider_budgets(groq_info)
            self.assertEqual(ov_req, 5500)
            self.assertEqual(ov_comp, 1200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
