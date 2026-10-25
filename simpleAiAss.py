# *standarad imports : 
import os
import json
import copy

# *Third party imports : 
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import platform

# *module imports : 
from Tools.tools import run_tool
from Tools.tools_schema import Tools
from Utils.utilities import log
from Storage.storage import (
    initialise_db,
    add_conversation,
    save_message,
    get_messages,
    get_conversations,
    get_user_facts,
)

load_dotenv()
EPOCHS = 10
# !Initialise the DB (SQLite)
initialise_db()

# !Start by allowing only the directory where the script is running
session_approved_dirs = { Path(os.getcwd()).resolve() }
HOME_DIR = os.path.expanduser("~")
CURRENT_OS = platform.system() # Returns "Darwin" (Mac), "Windows", or "Linux"

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

messages = []
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are an autonomous AI research agent. "
        "For complex queries, first output a brief step-by-step plan of which tools you will use. "
        "If a task requires multiple steps, use the output of one tool as the input for your next tool call. "
        "If a tool returns an error or empty results, DO NOT give up. Self-Reflect on why it failed, alter your parameters, and call the tool again. "
        "Call getCurrentTime for the time, get_weather for weather, and get_user_city if the user's city is missing. "
        "Call web_search whenever the user asks for the latest, newest, current, or most recent information. "
        "Construct a web_search query that asks for the latest result without assuming an outdated year. Compare all returned results to find the newest confirmed data, base your answer on these results, include source URLs, and state if results are conflicting. "
        "Use save_user_facts_to_db whenever the user reveals a personal preference, name, or permanent location. "
        f"You have access to the local file system. The user's home directory is {HOME_DIR}. "
        "Call list_directory to explore folders or find file names. "
        "Call read_file to inspect a specific file, and write_file to create or modify files. "
        "If asked to interact with folders like Desktop or Documents, construct the absolute path starting with the home directory. If a file name is not specified, always use list_directory first. "
        "CRITICAL INSTRUCTION: Never output raw JSON or function call payloads in your regular text response. If you need to use a tool, use the official tool calling format natively. If conversing with the user, use natural language only."

        "You operate in a strict dynamic sandbox. If you attempt to access a file or folder "
        "outside the initially approved project directory, the system will pause and ask the user for permission. "
        "If you receive an 'Access denied' error from a tool, politely ask the user in natural language "
        "where they would like you to save or read the file instead."
        "You can execute terminal commands using execute_command, but the user must manually approve each command."
        f"The user is running {CURRENT_OS}. When using the execute_command tool, "
        f"ensure you write standard shell commands appropriate for {CURRENT_OS}."

        "TOOL EXECUTION RULES:"
        "1. NEVER announce or explain what you are going to do before doing it. "
        "2. If a user asks you to perform an action (like running a command or writing a file), immediately output the tool call."
        "3. Do NOT output conversational text like 'I will run this command now' or 'Let me check that for you.'"
        "4. Execute the tool first, read the result, and ONLY provide conversational text to the user once the final task is complete."

        "REPORTING RULES:"
        "If a task requires multiple tool calls, your final text response MUST include an 'Execution Audit'. You must explicitly state:"
        "1. What you tried first."
        "2. Any exact error messages you received."
        "3. How you corrected it to get the final result."
    )
}

# ?Asking the user whether to start a new conversation or continue with the existing one

while True:
    print("1.Start a new conversation")
    print("2.Continue a previous conversation")

    #.strip() to remove the leading and the trailing white spaces
    user_choice = input("Enter your choice as a number (1 or 2): ").strip()

    if user_choice == "1":
        title = input("Enter the title for the new conversation : ").strip()

        conversation_id = add_conversation(title or "New Conversation")
        break

    elif user_choice == "2":
        conversations = get_conversations()
        if not conversations:
            print("No existing conversations, Starting a new one")
            title = input("Enter the title for the new conversation : ").strip()
            conversation_id = add_conversation(title or "New Conversation")
            break
        else:
            for conv in conversations:
                conv_id, title, created_at = conv  # unpack here
                print(f"conv_id : {conv_id}, title : {title}")

            valid_ids = {conv[0] for conv in conversations}

            while True:
                entered_id = input(
                    "Enter the id of the conversation you would like to continue : "
                ).strip()

                try:
                    selected_id = int(entered_id)
                except ValueError:
                    print("Please enter a valid numeric ID.")
                    continue

                if selected_id in valid_ids:
                    conversation_id = selected_id
                    break

                print("That conversation ID does not exist. Please try again.")

            break

    else:
        print("Invalid choice. Please enter 1 or 2.")


# 1.Retrieve the flat list of strings from the database
user_facts = get_user_facts()

if user_facts:
    formatted_memory = "\n\nCRITICAL USER FACTS (Long-Term Memory):\n"
    # Unpack both the ID and the fact text
    for fact_id, fact_text in user_facts:
        formatted_memory += f"- [ID: {fact_id}] {fact_text}\n"
    
    SYSTEM_PROMPT["content"] += formatted_memory

messages.append(SYSTEM_PROMPT)

# using .extend() as get_messages return a list so .append() is not suitable here
messages.extend(get_messages(conversation_id))


def enforce_context_budget(messages, max_chars=16000, history_reserve=4000, aggressive_cap=500):
    """
    Enforces a token budget using character limits.
    - Keeps atomic tool blocks (assistant tool-call + tool responses) together.
    - Aggressively truncates old tool outputs to preserve chat history.
    - Defensively caps the newest tool output to ensure history is never erased.
    """
    if not messages:
        return []

    system_prompt = messages[0]
    
    def get_size(msgs):
        return len(str(msgs))
        
    # If we are under budget, return as-is
    if get_size(messages) <= max_chars:
        return messages
        
    kept_messages = []
    # Account for system prompt size from the start
    current_chars = get_size(system_prompt)
    i = len(messages) - 1
    
    # Flag to ensure we treat the very latest interaction differently
    is_latest_turn = True 
    
    while i > 0:
        # Deep copy to prevent permanently mutating your database array
        msg = copy.deepcopy(messages[i])
        
        # --- 1. ATOMIC TOOL BLOCK HANDLING ---
        if msg.get("role") == "tool":
            block = [msg]
            i -= 1
            
            # Grab all adjacent tool responses
            while i > 0 and messages[i].get("role") == "tool":
                block.insert(0, copy.deepcopy(messages[i]))
                i -= 1
                
            # Grab the triggering assistant message
            if i > 0 and messages[i].get("role") == "assistant" and messages[i].get("tool_calls"):
                block.insert(0, copy.deepcopy(messages[i]))
                i -= 1
            else:
                # Discard orphaned tool calls without an assistant call
                continue
                
            # --- 2. TRUNCATION & COMPRESSION ---
            for b_msg in block:
                if b_msg.get("role") == "tool" and isinstance(b_msg.get("content"), str):
                    content = b_msg["content"]
                    
                    if not is_latest_turn:
                        # Aggressive compression for OLD tools
                        if len(content) > aggressive_cap:
                            b_msg["content"] = content[:aggressive_cap] + "\n...[Content truncated to preserve chat history]..."
                    else:
                        # Defensive cap for the CURRENT tool
                        # Leaves room for the user prompt and previous history
                        max_current_tool_size = max_chars - history_reserve 
                        if len(content) > max_current_tool_size:
                            b_msg["content"] = content[:max_current_tool_size] + "\n...[Truncated: Tool output too large for budget]..."
            
            block_size = get_size(block)
            
            # If adding this block STILL breaks the budget, stop slicing
            if current_chars + block_size > max_chars and kept_messages:
                break 
                
            kept_messages = block + kept_messages
            current_chars += block_size
            # Any previous tool block moving backwards is an old tool block
            is_latest_turn = False
            continue
            
        # --- 3. NORMAL MESSAGE HANDLING (User/Assistant Text) ---
        msg_size = get_size(msg)
        if current_chars + msg_size > max_chars and kept_messages:
            break # Stop if adding this message breaks the budget
            
        kept_messages.insert(0, msg)
        current_chars += msg_size
        i -= 1
        
        # Once we pass a user or text-only assistant message, older tools are no longer the "latest"
        if msg.get("role") in ["user", "assistant"]:
            is_latest_turn = False
            
    return [system_prompt] + kept_messages


print(f"Opened conversation {conversation_id}. Type 'exit' or 'bye' to quit.")

while True:
    user_input = input("User: ").strip()
    if user_input.lower() in ["exit", "quit", "bye"]:
        break
    if not user_input:
        continue

    user_message = {
         "role" : "user",
        "content" : user_input,
    }
    messages.append(user_message)
    save_message(conversation_id, user_message)

    # ?Check if model wants to call any tool and if yes, call the tool and add the tool response to the messages array so that the model can use it in the next turn
    for iteration in range(EPOCHS):
        # --- SLIDING WINDOW ENFORCEMENT ---
        # Cap at 16,000 chars (approx 4k tokens). 
        # Reserves 4,000 chars strictly for conversation history.
        budgeted_messages = enforce_context_budget(messages, max_chars=16000)

        response = client.chat.completions.create(
            model = os.getenv("MODEL"),
            messages = budgeted_messages,
            tools = Tools,
        )

        model_response = response.choices[0].message

        if not model_response.tool_calls:
            content = model_response.content or ""
            assistant_message = {
                "role": "assistant",
                "content": content,
            }

            print("🤖 Assistant:", content if content else "(No response)")

            messages.append(assistant_message)
            save_message(conversation_id, assistant_message)

            break

        log(f"Model requested tool calls : {len(model_response.tool_calls)}")

        # ?Append the model response to the messages 
        
        # *model_dump() is used to convert the model's response object into a Python dictionary.
        assistant_tool_message = model_response.model_dump(exclude_none=True)
        messages.append(assistant_tool_message)
        save_message(conversation_id, assistant_tool_message)

        for tool_call in model_response.tool_calls:
            func_name = tool_call.function.name
            call_id = tool_call.id or f"call_{iteration}"

            # Safely parse tool arguments
            try:
                args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                if not isinstance(args, dict):
                    args = {}
            except Exception:
                args = {}

            # !check the files Agent wants to access should be given access or not
            if func_name in ["read_file", "write_file", "list_directory"]:
                # Get whichever path parameter the tool uses
                target = args.get("file_path") or args.get("directory_path")

                if not target:
                    error_msg = f"Error: Tool '{func_name}' requires a valid path argument ('file_path' or 'directory_path')."
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(error_msg),
                    }
                    messages.append(tool_message)
                    save_message(conversation_id, tool_message)
                    continue

                # Resolve the absolute path
                expanded_target = Path(os.path.expanduser(target)).resolve()

                # Check if the target is inside any currently approved directory
                is_approved = any(expanded_target.is_relative_to(approved) for approved in session_approved_dirs)

                if not is_approved:
                    print(f"\nSECURITY ALERT: Agent wants to access outside the sandbox: {expanded_target}")

                    approval = input("Grant read/write access to this directory for this session? (y/n): ").strip().lower()

                    if approval == "y":
                        # If target is an existing directory approve it; otherwise approve its parent folder
                        if expanded_target.is_dir():
                            target_directory = expanded_target
                        else:
                            target_directory = expanded_target.parent

                        session_approved_dirs.add(target_directory)
                        print(f"✅ Access granted to {target_directory}")

                    else:
                        # Feed the rejection back to the model
                        func_response = f"Access denied by user to path: {target}. You must ask the user for a different path."
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": json.dumps(func_response),
                        }
                        messages.append(tool_message)
                        save_message(conversation_id, tool_message)
                        # Skip execution and let the model read the denial
                        continue

            elif func_name == "execute_command":
                # !check if the model is trying to run the correct command and not something harmful
                command = args.get("command", "")
                
                print(f"\nTERMINAL ALERT: Agent wants to execute: `{command}`")
                approval = input("Allow execution? (y/n): ").strip().lower()
                
                if approval != "y":
                    func_response = "Access denied by user. Do not attempt to run this command again."
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(func_response),
                    }
                    messages.append(tool_message)
                    save_message(conversation_id, tool_message)
                    # Skip execution
                    continue

            # ?Call the tool and get the response
            func_response = run_tool(tool_call)

            # ?Append the tool response to the messages array so that the model can use it in the next turn
            content_str = func_response if isinstance(func_response, str) else json.dumps(func_response)
            tool_message = {
                "role" : "tool",
                "tool_call_id" : call_id,
                "content" : content_str,
            }

            messages.append(tool_message)
            save_message(conversation_id, tool_message)

    else:
        error_msg = f"Agent reached maximum execution limit of {EPOCHS} steps without completing the task."
        print(f"🛑 Error: {error_msg}")
        
        # Save the system failure to context so the next prompt has awareness
        failure_message = {
            "role": "system",
            "content": error_msg
        }
        messages.append(failure_message)
        save_message(conversation_id, failure_message)
    

print("conversation ended.")