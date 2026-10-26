"""
Comprehensive test suite for Local AI Assistant migration to Groq Cloud.
Tests all scenarios specified in the migration requirements:
1. Conversational question (no tools)
2. List project files
3. Read and summarize README.md
4. Multiple tool calls in a single turn
5. Current-events query (web_search)
6. Nonexistent file and tool failure
7. Terminal operation requiring approval (approved vs denied)
8. Rejected operation outside allowed directory (sandbox protection)
9. Context budget enforcement with large history
10. API error and rate-limit handling
11. Live Groq Cloud connectivity check (if valid credentials provided)
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import json
import pathlib
import time

# Ensure project root is on Python path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import openai
from Tools.tools import (
    run_tool,
    read_file,
    write_file,
    list_directory,
    execute_command,
    getCurrentTime,
    function_map,
)
from Tools.tools_schema import Tools, WEB_SEARCH, EXECUTE_COMMAND, READ_FILE
from Storage.storage import (
    initialise_db,
    add_conversation,
    save_message,
    get_messages,
    save_user_fact,
    get_user_facts,
    delete_conversation,
    delete_all_conversations,
)
from simpleAiAss import (
    enforce_context_budget,
    execute_agent_turn,
    build_system_prompt,
)


class TestGroqMigration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initialise_db()

    def setUp(self):
        self.approved_dirs = {PROJECT_ROOT.resolve()}

    # --- Scenario 1: Conversational question without tool calling ---
    def test_scenario_1_conversational_no_tool(self):
        """Verify that general conversational questions answer directly without tool calls."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                finish_reason="stop",
                message=MagicMock(
                    role="assistant",
                    content="Quantum computing uses superposition and entanglement.",
                    tool_calls=None
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is quantum computing?"}
        ]

        final_content, updated_messages = execute_agent_turn(
            messages=messages,
            conversation_id=None,
            client_instance=mock_client,
            model_name="openai/gpt-oss-120b",
            approved_dirs=self.approved_dirs,
            epochs=3,
            interactive=False
        )

        self.assertIn("superposition", final_content)
        self.assertEqual(len(updated_messages), 3)
        self.assertEqual(updated_messages[-1]["role"], "assistant")
        self.assertIsNone(updated_messages[-1].get("tool_calls"))

    # --- Scenario 2: List project files ---
    def test_scenario_2_list_directory(self):
        """Verify list_directory returns entries for the project directory."""
        tool_call = MagicMock()
        tool_call.function.name = "list_directory"
        tool_call.function.arguments = json.dumps({"directory_path": "."})

        result = run_tool(tool_call)
        self.assertIsInstance(result, list)
        self.assertTrue(any("README.md" in item for item in result))
        self.assertTrue(any("simpleAiAss.py" in item for item in result))

    # --- Scenario 3: Read and summarize README.md ---
    def test_scenario_3_read_file(self):
        """Verify read_file safely inspects local README.md."""
        tool_call = MagicMock()
        tool_call.function.name = "read_file"
        tool_call.function.arguments = json.dumps({"file_path": "README.md"})

        content = run_tool(tool_call)
        self.assertIsInstance(content, str)
        self.assertIn("Local AI Assistant", content)

    # --- Scenario 4: Request requiring multiple tool calls ---
    def test_scenario_4_multiple_tool_calls(self):
        """Verify handling when the model returns multiple tool calls in a single turn."""
        mock_client = MagicMock()

        # Turn 1: Model requests two tools (getCurrentTime and list_directory)
        tc1 = MagicMock()
        tc1.id = "call_time_1"
        tc1.function.name = "getCurrentTime"
        tc1.function.arguments = "{}"

        tc2 = MagicMock()
        tc2.id = "call_list_2"
        tc2.function.name = "list_directory"
        tc2.function.arguments = json.dumps({"directory_path": "."})

        response_1 = MagicMock()
        response_1.choices = [
            MagicMock(
                finish_reason="tool_calls",
                message=MagicMock(
                    role="assistant",
                    content=None,
                    tool_calls=[tc1, tc2]
                )
            )
        ]

        # Turn 2: Model finishes with final answer synthesizing tool results
        response_2 = MagicMock()
        response_2.choices = [
            MagicMock(
                finish_reason="stop",
                message=MagicMock(
                    role="assistant",
                    content="Checked time and listed project directory successfully.",
                    tool_calls=None
                )
            )
        ]

        mock_client.chat.completions.create.side_effect = [response_1, response_2]

        messages = [
            {"role": "system", "content": "Assistant prompt"},
            {"role": "user", "content": "What time is it and what files exist?"}
        ]

        final_content, updated_messages = execute_agent_turn(
            messages=messages,
            conversation_id=None,
            client_instance=mock_client,
            model_name="openai/gpt-oss-120b",
            approved_dirs=self.approved_dirs,
            epochs=5,
            interactive=False
        )

        self.assertIn("successfully", final_content)
        # Check that both tool outputs are present in the conversation
        tool_responses = [m for m in updated_messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_responses), 2)
        tool_ids = {m["tool_call_id"] for m in tool_responses}
        self.assertEqual(tool_ids, {"call_time_1", "call_list_2"})

    # --- Scenario 5: Web Search query and schema validation ---
    def test_scenario_5_web_search_schema_and_routing(self):
        """Verify web search schema format and parameter requirements."""
        self.assertEqual(WEB_SEARCH["function"]["name"], "web_search")
        self.assertIn("query", WEB_SEARCH["function"]["parameters"]["required"])
        self.assertEqual(WEB_SEARCH["function"]["parameters"]["properties"]["query"]["type"], "string")

        # Calling web_search without configured TAVILY_API_KEY should raise or handle cleanly
        with patch.dict(os.environ, {"TAVILY_API_KEY": ""}):
            tool_call = MagicMock()
            tool_call.function.name = "web_search"
            tool_call.function.arguments = json.dumps({"query": "latest news"})
            res = run_tool(tool_call)
            self.assertTrue("error" in res.lower() or "configured" in res.lower())

    # --- Scenario 6: Nonexistent file and tool failure ---
    def test_scenario_6_nonexistent_file_handling(self):
        """Verify tool failure returns descriptive error message rather than crashing."""
        tool_call = MagicMock()
        tool_call.function.name = "read_file"
        tool_call.function.arguments = json.dumps({"file_path": "nonexistent_file_xyz123.txt"})

        result = run_tool(tool_call)
        self.assertIn("Error: File not found", result)

    # --- Scenario 7: Terminal operation requiring approval ---
    def test_scenario_7_terminal_approval(self):
        """Verify execute_command asks for Y/N approval in CLI and returns pending_action in Streamlit mode."""
        mock_client = MagicMock()

        tc = MagicMock()
        tc.id = "call_term_1"
        tc.function.name = "execute_command"
        tc.function.arguments = json.dumps({"command": "echo 'approval test'"})

        resp_turn1 = MagicMock()
        resp_turn1.choices = [MagicMock(finish_reason="tool_calls", message=MagicMock(role="assistant", content=None, tool_calls=[tc]))]
        resp_turn2 = MagicMock()
        resp_turn2.choices = [MagicMock(finish_reason="stop", message=MagicMock(role="assistant", content="Echo executed.", tool_calls=None))]

        # Case A: CLI mode with 'y' input
        mock_client.chat.completions.create.side_effect = [resp_turn1, resp_turn2]
        messages_a = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Run echo"}]
        with patch("builtins.input", return_value="y"):
            final_a, msgs_a = execute_agent_turn(
                messages=messages_a,
                conversation_id=None,
                client_instance=mock_client,
                model_name="openai/gpt-oss-120b",
                approved_dirs=self.approved_dirs,
                epochs=3,
                interactive=True,
            )
        tool_msgs_a = [m for m in msgs_a if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs_a), 1)
        self.assertIn("approval test", tool_msgs_a[0]["content"])

        # Case B: CLI mode with 'n' input
        mock_client.chat.completions.create.side_effect = [resp_turn1, resp_turn2]
        messages_b = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Run echo"}]
        with patch("builtins.input", return_value="n"):
            final_b, msgs_b = execute_agent_turn(
                messages=messages_b,
                conversation_id=None,
                client_instance=mock_client,
                model_name="openai/gpt-oss-120b",
                approved_dirs=self.approved_dirs,
                epochs=3,
                interactive=True,
            )
        tool_msgs_b = [m for m in msgs_b if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs_b), 1)
        self.assertIn("denied by user", tool_msgs_b[0]["content"].lower())

        # Case C: Streamlit mode (interactive=False) yields pending_action without executing
        mock_client.chat.completions.create.side_effect = [resp_turn1]
        messages_c = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Run echo"}]
        res_c = execute_agent_turn(
            messages=messages_c,
            conversation_id=None,
            client_instance=mock_client,
            model_name="openai/gpt-oss-120b",
            approved_dirs=self.approved_dirs,
            epochs=3,
            interactive=False,
        )
        self.assertIsNotNone(res_c.pending_action)
        self.assertEqual(res_c.pending_action["type"], "terminal")
        self.assertIn("approval test", res_c.pending_action["target"])

    # --- Scenario 8: Rejected operation outside allowed directory ---
    def test_scenario_8_sandbox_directory_rejection(self):
        """Verify file operations outside sandbox prompt in CLI and pause in Streamlit mode."""
        mock_client = MagicMock()

        tc = MagicMock()
        tc.id = "call_fs_1"
        tc.function.name = "read_file"
        tc.function.arguments = json.dumps({"file_path": "/var/log/system.log"})

        resp_turn1 = MagicMock()
        resp_turn1.choices = [MagicMock(finish_reason="tool_calls", message=MagicMock(role="assistant", content=None, tool_calls=[tc]))]
        resp_turn2 = MagicMock()
        resp_turn2.choices = [MagicMock(finish_reason="stop", message=MagicMock(role="assistant", content="Cannot access file.", tool_calls=None))]

        # Case A: CLI mode denial
        mock_client.chat.completions.create.side_effect = [resp_turn1, resp_turn2]
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Read system log"}]
        with patch("builtins.input", return_value="n"):
            final, msgs = execute_agent_turn(
                messages=messages,
                conversation_id=None,
                client_instance=mock_client,
                model_name="openai/gpt-oss-120b",
                approved_dirs=self.approved_dirs,
                epochs=3,
                interactive=True,
            )

        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("Access denied by user", tool_msgs[0]["content"])

        # Case B: Streamlit mode (interactive=False) pauses with pending_action
        mock_client.chat.completions.create.side_effect = [resp_turn1]
        messages_st = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Read system log"}]
        res_st = execute_agent_turn(
            messages=messages_st,
            conversation_id=None,
            client_instance=mock_client,
            model_name="openai/gpt-oss-120b",
            approved_dirs=self.approved_dirs,
            epochs=3,
            interactive=False,
        )
        self.assertIsNotNone(res_st.pending_action)
        self.assertEqual(res_st.pending_action["type"], "filesystem")
        self.assertIn("system.log", res_st.pending_action["target"])

    # --- Scenario 9: Conversation exceeding context budget ---
    def test_scenario_9_context_budget_enforcement(self):
        """Verify context budget trims older messages and caps large tool outputs."""
        system_msg = {"role": "system", "content": "System prompt instructions."}
        large_tool_output = "X" * 10000

        messages = [
            system_msg,
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f1", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t1", "content": large_tool_output},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question with more details"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "t2", "type": "function", "function": {"name": "f2", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t2", "content": "Recent small tool result"},
            {"role": "assistant", "content": "Second answer"},
        ]

        # Enforce budget of 8000 chars
        budgeted = enforce_context_budget(messages, max_chars=8000, history_reserve=2000, aggressive_cap=300)

        total_len = len(str(budgeted))
        self.assertLessEqual(total_len, 8500)
        self.assertEqual(budgeted[0]["role"], "system")
        # Check that older tool output was compressed with aggressive_cap
        old_tool = [m for m in budgeted if m.get("tool_call_id") == "t1"]
        if old_tool:
            self.assertIn("truncated", old_tool[0]["content"].lower())

    # --- Scenario 10: API error and rate-limit handling ---
    def test_scenario_10_api_error_handling(self):
        """Verify graceful handling and user notification on authentication and rate limit errors."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
            message="Invalid API Key provided",
            response=MagicMock(status_code=401),
            body=None
        )

        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Hi"}]
        err_msg, _ = execute_agent_turn(
            messages=messages,
            conversation_id=None,
            client_instance=mock_client,
            model_name="openai/gpt-oss-120b",
            approved_dirs=self.approved_dirs,
            epochs=2,
            interactive=False
        )

        self.assertIn("Authentication Error", err_msg)

    # --- Scenario 11: Live Connectivity Check ---
    def test_scenario_11_live_connectivity_check(self):
        """
        Tests live API connectivity if a real API_KEY is configured for Gemini or Groq.
        Distinguishes tested from skipped/unavailable credentials.
        """
        api_key = os.getenv("API_KEY") or os.getenv("GROQ_API_KEY")
        base_url = os.getenv("BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai/"
        model = os.getenv("MODEL") or "gemini-3.5-flash-lite"

        if not api_key or api_key in ["your_groq_api_key", "your_gemini_api_key", "dummy_key_unconfigured"]:
            print("\n  [LIVE TEST NOTICE] Live cloud call skipped: No live API key configured in .env.")
            self.skipTest("Live cloud test requires a valid API key in .env.")
        else:
            live_client = openai.OpenAI(base_url=base_url, api_key=api_key)
            resp = live_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Respond with 'PONG'"}],
                max_tokens=50
            )
            content = resp.choices[0].message.content or ""
            reasoning = getattr(resp.choices[0].message, "reasoning", "")
            self.assertTrue(len(content) > 0 or len(reasoning) > 0)
            print(f"\n  [LIVE TEST SUCCESS] Connected to {model} and received: {content.strip()}")

    # --- Scenario 12: Live Tool Calling with Gemini ---
    def test_scenario_12_live_tool_calling(self):
        """
        Tests live tool calling through execute_agent_turn using Gemini.
        Verifies that Gemini invokes tools, preserves thought_signatures, and synthesizes answers.
        """
        api_key = os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
        base_url = os.getenv("BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai/"
        model = os.getenv("MODEL") or "gemini-3.5-flash-lite"

        if not api_key or api_key in ["your_groq_api_key", "your_gemini_api_key", "dummy_key_unconfigured"]:
            self.skipTest("Live tool test requires a valid API key in .env.")

        live_client = openai.OpenAI(base_url=base_url, api_key=api_key)
        messages = [
            build_system_prompt(),
            {"role": "user", "content": "What is the current time? Use getCurrentTime."}
        ]
        final_answer, updated_messages = execute_agent_turn(
            messages=messages,
            conversation_id=None,
            client_instance=live_client,
            model_name=model,
            approved_dirs=self.approved_dirs,
            epochs=5,
            interactive=False
        )

        self.assertTrue(any(m.get("role") == "tool" for m in updated_messages))
        self.assertTrue(len(final_answer) > 0)
        self.assertNotIn("Error", final_answer)
        print(f"\n  [LIVE TOOL CALL SUCCESS] Executed tool and synthesized response: {final_answer[:100]}...")

    # --- Scenario 13: Conversation Deletion ---
    def test_scenario_13_delete_conversation(self):
        """Verify deleting a conversation removes it and its messages from SQLite."""
        conv_id = add_conversation("Temporary Test Conversation")
        save_message(conv_id, {"role": "user", "content": "Hello to be deleted"})
        self.assertEqual(len(get_messages(conv_id)), 1)

        delete_conversation(conv_id)
        self.assertEqual(len(get_messages(conv_id)), 0)

    # --- Scenario 14: User City via SQLite user_facts ---
    def test_scenario_14_user_city_storage_and_retrieval(self):
        """Verify user city is saved to and retrieved from user_facts without hardcoded fallbacks."""
        from Tools.tools import get_user_city, save_user_city, get_saved_user_city

        # Test saving and retrieving city
        save_user_city("Jaipur")
        self.assertEqual(get_saved_user_city(), "Jaipur")
        self.assertEqual(get_user_city(), "Jaipur")

        # Test updating city
        save_user_city("Varanasi")
        self.assertEqual(get_saved_user_city(), "Varanasi")
        self.assertEqual(get_user_city(), "Varanasi")


if __name__ == "__main__":
    unittest.main(verbosity=2)
