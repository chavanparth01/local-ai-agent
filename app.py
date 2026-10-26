"""
Streamlit Web UI for Local AI Assistant
ChatGPT-inspired modern design with custom sidebar chat list, responsive layout,
and refined dark theme aesthetic.
"""

import os
import json
import html
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()
os.environ["ASSISTANT_MODE"] = "web"

from Storage.storage import (
    initialise_db,
    add_conversation,
    get_conversations,
    get_messages,
    save_message,
    delete_conversation,
    delete_all_conversations,
    update_conversation_title,
    get_user_facts,
    delete_user_fact_db,
    save_user_fact,
)
from simpleAiAss import (
    execute_agent_turn,
    build_system_prompt,
    client,
    MODEL,
    session_approved_dirs,
    MAX_REQUEST_TOKENS,
    MAX_COMPLETION_TOKENS,
    EPOCHS,
    provider_info,
)
from Tools.tools_schema import Tools
from Tools.tools import get_saved_user_city, save_user_city, run_tool
from Utils.utilities import estimate_request_tokens

# 1. Initialize SQLite Database
initialise_db()

# 2. Page Configuration
st.set_page_config(
    page_title="Local AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. Modern ChatGPT-Inspired Design System & Responsive CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Typography & Palette */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #e2e8f0;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Remove excess top padding on main container */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 5rem !important;
        max-width: 900px !important;
        margin: 0 auto !important;
    }

    /* Responsive Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem !important;
        padding-left: 0.9rem !important;
        padding-right: 0.9rem !important;
    }

    /* Sidebar Section Headers */
    .sidebar-section-header {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        margin: 1.2rem 0 0.5rem 0.2rem;
    }

    /* Assistant Status Card */
    .status-card {
        background: rgba(22, 27, 38, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin-bottom: 1rem;
    }
    .status-title {
        font-size: 0.88rem;
        font-weight: 600;
        color: #f1f5f9;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .status-subtitle {
        font-size: 0.76rem;
        color: #94a3b8;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    .status-badge {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 6px;
        background: rgba(16, 185, 129, 0.12);
        color: #10b981;
        font-size: 0.72rem;
        font-weight: 600;
        margin-top: 6px;
    }

    /* Sidebar Conversation Item */
    .conv-item-active {
        background: rgba(59, 130, 246, 0.14) !important;
        border-left: 3px solid #3b82f6 !important;
        border-radius: 6px !important;
    }

    /* Main Chat Top Bar */
    .chat-topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 0.2rem 1rem 0.2rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
        margin-bottom: 1.5rem;
    }
    .chat-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .chat-badge {
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 4px 9px;
        border-radius: 20px;
        color: #94a3b8;
    }

    /* User Message Bubble - Right Side */
    .chat-row-user {
        display: flex;
        justify-content: flex-end;
        align-items: flex-start;
        margin: 1.4rem 0;
        width: 100%;
    }
    .user-bubble {
        background: #1e2433;
        border: 1px solid rgba(255, 255, 255, 0.09);
        color: #f8fafc !important;
        padding: 0.95rem 1.35rem;
        border-radius: 18px 18px 4px 18px;
        max-width: 78%;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        font-size: 0.96rem;
        line-height: 1.6;
        word-wrap: break-word;
    }
    .user-bubble-header {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #94a3b8;
        margin-bottom: 0.35rem;
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 5px;
    }
    .user-bubble-content {
        color: #f8fafc !important;
        white-space: pre-wrap;
    }

    /* Assistant Message Enhancement */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
    div[data-testid="stChatMessage"]:has([aria-label="chat message from assistant"]) {
        background-color: rgba(22, 27, 38, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        padding: 1.1rem 1.35rem !important;
        margin: 1.2rem 0 !important;
        max-width: 95% !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.18) !important;
        line-height: 1.65 !important;
    }

    /* Modern Tool Badge */
    .tool-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(56, 189, 248, 0.09);
        border: 1px solid rgba(56, 189, 248, 0.22);
        color: #38bdf8;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 6px;
    }

    /* Security Permission Authorization Card */
    .security-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    .security-header {
        font-size: 1rem;
        font-weight: 700;
        color: #fbbf24;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.5rem;
    }
    .security-body {
        font-size: 0.88rem;
        color: #cbd5e1;
        margin-bottom: 1rem;
    }

    /* Primary buttons (New chat, Active chat) override to cohesive dark theme */
    button[kind="primary"],
    .stButton>button[kind="primary"] {
        background: #2563eb !important;
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25) !important;
    }
    button[kind="primary"]:hover,
    .stButton>button[kind="primary"]:hover {
        background: #1d4ed8 !important;
        background-color: #1d4ed8 !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
    }

    /* Sidebar Conversation Row Container - Force single row, no wrapping, clean alignment */
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        gap: 4px !important;
        margin-bottom: 3px !important;
    }

    /* Column 1: Chat Title Button - Takes all remaining width, truncates cleanly */
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child {
        flex: 1 1 auto !important;
        min-width: 0 !important;
        padding: 0 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child button {
        height: 34px !important;
        min-height: 34px !important;
        max-height: 34px !important;
        padding: 0 10px !important;
        margin: 0 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        border-radius: 7px !important;
        font-size: 0.85rem !important;
        width: 100% !important;
    }

    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child button p {
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        width: 100% !important;
        text-align: left !important;
    }

    /* Column 2 & 3: Rename and Delete Action Icons - Fixed width 32px, never squish */
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2),
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(3) {
        flex: 0 0 32px !important;
        width: 32px !important;
        min-width: 32px !important;
        max-width: 32px !important;
        padding: 0 !important;
    }

    /* Buttons inside action columns */
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) button,
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(3) button {
        width: 32px !important;
        height: 34px !important;
        min-width: 32px !important;
        max-width: 32px !important;
        min-height: 34px !important;
        max-height: 34px !important;
        padding: 0 !important;
        margin: 0 !important;
        border-radius: 7px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        background: rgba(255, 255, 255, 0.04) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 0.85rem !important;
        color: #94a3b8 !important;
    }

    /* Hide the popover chevron SVG completely so only the icon shows */
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) button svg {
        display: none !important;
    }

    /* Hover effects for Rename and Delete */
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) button:hover {
        background: rgba(56, 189, 248, 0.15) !important;
        border-color: rgba(56, 189, 248, 0.35) !important;
        color: #38bdf8 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(3) button:hover {
        background: rgba(239, 68, 68, 0.15) !important;
        border-color: rgba(239, 68, 68, 0.35) !important;
        color: #f87171 !important;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-top: 1rem !important;
        }
        .user-bubble {
            max-width: 95% !important;
            padding: 0.85rem 1.1rem !important;
        }
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
            max-width: 100% !important;
            padding: 0.9rem 1rem !important;
        }
        .chat-title {
            font-size: 1.05rem !important;
        }
        .stButton>button {
            min-height: 44px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# 4. Sidebar: Control Panel & ChatGPT-Style Chat List
with st.sidebar:
    # 4.1 New Chat Button
    if st.button("➕ New chat", use_container_width=True, type="primary"):
        new_id = add_conversation(f"New Chat {len(get_conversations()) + 1}")
        st.session_state["selected_conv_id"] = new_id
        st.session_state["messages"] = [build_system_prompt()]
        st.session_state["pending_action"] = None
        st.rerun()

    st.markdown('<div class="sidebar-section-header">ASSISTANT STATUS</div>', unsafe_allow_html=True)

    # 4.2 Assistant Status & Model Card
    provider_name = provider_info.get("provider", "LLM Engine")
    is_cloud = provider_info.get("is_cloud", True)
    status_dot = "🟢" if is_cloud else "🔵"

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-title">{status_dot} {provider_name}</div>
            <div class="status-subtitle">{MODEL}</div>
            <div class="status-badge">15 RPM • 250K TPM • 500 RPD</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4.3 Context Budget & Guardrails Expander
    with st.expander("📊 Token Budget & Limits", expanded=False):
        c_b1, c_b2 = st.columns(2)
        c_b1.metric("Max Request", f"{MAX_REQUEST_TOKENS // 1000}k")
        c_b2.metric("Completion", f"{MAX_COMPLETION_TOKENS // 1000}k")
        st.caption("Sliding window ensures complete atomic tool sequences without context overflow.")

    # 4.4 User Location Preference
    st.markdown('<div class="sidebar-section-header">LOCATION SETTINGS</div>', unsafe_allow_html=True)
    current_saved_city = get_saved_user_city() or ""
    new_city_input = st.text_input(
        "Current City",
        value=current_saved_city,
        placeholder="Enter city (e.g. Mumbai, Prayagraj)",
        label_visibility="collapsed",
        help="Stored in SQLite user_facts and automatically supplied to local tools."
    )
    if new_city_input and new_city_input.strip() != current_saved_city:
        save_user_city(new_city_input.strip())
        st.toast(f"📍 City set to {new_city_input.strip()}")
        st.rerun()

    # 4.5 Sidebar Chat List (Create, Select, Rename, Delete)
    st.markdown('<div class="sidebar-section-header">RECENT CONVERSATIONS</div>', unsafe_allow_html=True)

    conversations = get_conversations()
    if not conversations:
        new_id = add_conversation("New Chat 1")
        st.session_state["selected_conv_id"] = new_id
        st.session_state["messages"] = [build_system_prompt()]
        conversations = get_conversations()

    active_conv_id = st.session_state.get("selected_conv_id")
    conv_ids = [c[0] for c in conversations]

    if active_conv_id not in conv_ids:
        active_conv_id = conversations[0][0]
        st.session_state["selected_conv_id"] = active_conv_id

    # Render each conversation item with Select, Rename popover, and Delete
    for conv in conversations:
        cid, title, created_at = conv
        is_active = (cid == active_conv_id)

        col_select, col_rename, col_delete = st.columns([0.76, 0.12, 0.12], gap="small")

        display_label = title if len(title) <= 22 else f"{title[:20]}..."
        button_label = f"💬 {display_label}" if not is_active else f"▶ {display_label}"

        with col_select:
            # Active chat has primary type, inactive has secondary
            if st.button(
                button_label,
                key=f"btn_conv_{cid}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                help=f"{title}\nCreated: {created_at}"
            ):
                if cid != active_conv_id:
                    st.session_state["selected_conv_id"] = cid
                    st.session_state["messages"] = [build_system_prompt()] + get_messages(cid)
                    st.session_state["pending_action"] = None
                    st.rerun()

        with col_rename:
            with st.popover("✏️", help=f"Rename '{title}'"):
                st.markdown(f"**Rename Chat #{cid}**")
                edit_title = st.text_input("New title:", value=title, key=f"rename_input_{cid}")
                if st.button("Save", key=f"save_rename_{cid}", use_container_width=True):
                    cleaned = edit_title.strip()
                    if cleaned and cleaned != title:
                        update_conversation_title(cid, cleaned)
                        st.toast("Title updated!")
                        st.rerun()

        with col_delete:
            if st.button("🗑️", key=f"del_conv_{cid}", help=f"Delete '{title}'"):
                delete_conversation(cid)
                remaining = get_conversations()
                if remaining:
                    st.session_state["selected_conv_id"] = remaining[0][0]
                    st.session_state["messages"] = [build_system_prompt()] + get_messages(remaining[0][0])
                else:
                    new_id = add_conversation("New Chat")
                    st.session_state["selected_conv_id"] = new_id
                    st.session_state["messages"] = [build_system_prompt()]
                st.session_state["pending_action"] = None
                st.toast("Chat deleted")
                st.rerun()

    # 4.6 Clear All Chats Action
    st.divider()
    with st.expander("⚙️ Storage Maintenance", expanded=False):
        if st.button("🧹 Clear All Chats", use_container_width=True, help="Wipe all conversation history from database"):
            st.session_state["confirm_clear_all"] = True

        if st.session_state.get("confirm_clear_all"):
            st.warning("Delete ALL conversations and messages?")
            c_conf1, c_conf2 = st.columns(2)
            if c_conf1.button("Confirm Wipe", type="primary", use_container_width=True):
                delete_all_conversations()
                new_id = add_conversation("New Chat 1")
                st.session_state["selected_conv_id"] = new_id
                st.session_state["messages"] = [build_system_prompt()]
                st.session_state["confirm_clear_all"] = False
                st.toast("All chats wiped!")
                st.rerun()
            if c_conf2.button("Cancel", use_container_width=True):
                st.session_state["confirm_clear_all"] = False
                st.rerun()

    # 4.7 Long-Term Facts Memory
    with st.expander("🧠 Persistent Memory Facts", expanded=False):
        user_facts = get_user_facts()
        if user_facts:
            for fid, ftext in user_facts:
                col_f1, col_f2 = st.columns([4.5, 1])
                col_f1.caption(f"• {ftext}")
                if col_f2.button("🗑️", key=f"del_fact_{fid}"):
                    delete_user_fact_db(fid)
                    st.rerun()
        else:
            st.caption("No persistent facts yet. Assistant automatically saves key preferences.")

# 5. Main Chat Area
current_conv_id = st.session_state.get("selected_conv_id")

# Fetch current conversation title
current_title = "Chat"
if conversations:
    match_title = [c[1] for c in conversations if c[0] == current_conv_id]
    if match_title:
        current_title = match_title[0]

# Chat Top Bar
st.markdown(
    f"""
    <div class="chat-topbar">
        <div class="chat-title">
            <span>💬</span> <span>{html.escape(current_title)}</span>
        </div>
        <div class="chat-badge">{MODEL}</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Ensure session messages are loaded
if "messages" not in st.session_state or not st.session_state["messages"]:
    db_msgs = get_messages(current_conv_id) if current_conv_id else []
    st.session_state["messages"] = [build_system_prompt()] + db_msgs


def group_messages_into_turns(messages):
    """Groups linear message history into neat user-assistant conversational turns."""
    turns = []
    current_turn = None

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue

        if role == "user":
            if current_turn:
                turns.append(current_turn)
            current_turn = {
                "user": msg.get("content", ""),
                "tool_executions": [],
                "assistant_content": ""
            }
        elif current_turn is not None:
            if role == "assistant":
                tool_calls = msg.get("tool_calls") or []
                for tc in tool_calls:
                    call_id = tc.get("id")
                    func_obj = tc.get("function") or {}
                    func_name = func_obj.get("name", "tool")
                    args = func_obj.get("arguments", "{}")
                    current_turn["tool_executions"].append({
                        "id": call_id,
                        "name": func_name,
                        "args": args,
                        "output": None
                    })
                content = msg.get("content")
                if content:
                    current_turn["assistant_content"] = (
                        current_turn["assistant_content"] + "\n\n" + content
                        if current_turn["assistant_content"] else content
                    )
            elif role == "tool":
                call_id = msg.get("tool_call_id")
                content = msg.get("content", "")
                matched = False
                for te in current_turn["tool_executions"]:
                    if te["id"] == call_id and te["output"] is None:
                        te["output"] = content
                        matched = True
                        break
                if not matched:
                    current_turn["tool_executions"].append({
                        "id": call_id,
                        "name": "tool_result",
                        "args": "",
                        "output": content
                    })
        else:
            if role == "assistant" and msg.get("content"):
                turns.append({
                    "user": None,
                    "tool_executions": [],
                    "assistant_content": msg.get("content")
                })

    if current_turn:
        turns.append(current_turn)

    return turns


# 6. Render Conversation History
turns = group_messages_into_turns(st.session_state["messages"])

if not turns:
    # Empty State: Friendly modern prompt cards
    st.markdown(
        """
        <div style="text-align: center; padding: 3rem 1rem 2rem 1rem;">
            <div style="font-size: 2.8rem; margin-bottom: 0.8rem;">🤖</div>
            <h2 style="font-weight: 700; color: #f8fafc; margin-bottom: 0.4rem;">How can I assist you today?</h2>
            <p style="color: #94a3b8; font-size: 0.95rem; max-width: 500px; margin: 0 auto 2rem auto;">
                Local AI Assistant with terminal commands, sandboxed filesystem access, real-time web search, and weather tools.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    for turn in turns:
        # Render User Message on Right
        if turn.get("user"):
            escaped_user = html.escape(turn["user"])
            st.markdown(
                f"""
                <div class="chat-row-user">
                    <div class="user-bubble">
                        <div class="user-bubble-header">
                            <span>You</span> <span>👤</span>
                        </div>
                        <div class="user-bubble-content">{escaped_user}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Render Assistant Response on Left
        has_tools = bool(turn.get("tool_executions"))
        has_content = bool(turn.get("assistant_content"))

        if has_tools or has_content:
            with st.chat_message("assistant", avatar="🤖"):
                if has_tools:
                    num_tools = len(turn["tool_executions"])
                    with st.expander(f"🛠️ Executed {num_tools} Tool Action{'s' if num_tools > 1 else ''}", expanded=False):
                        for exec_item in turn["tool_executions"]:
                            st.markdown(f'<div class="tool-chip">🔧 {html.escape(exec_item["name"])}</div>', unsafe_allow_html=True)
                            if exec_item.get("args") and exec_item["args"] != "{}":
                                st.caption(f"**Arguments:** `{exec_item['args']}`")
                            if exec_item.get("output"):
                                out_str = exec_item["output"]
                                if len(out_str) > 2000:
                                    out_str = out_str[:2000] + "\n...[truncated for display]..."
                                st.code(out_str, language="text")
                            st.divider()

                if has_content:
                    st.markdown(turn["assistant_content"])
                elif not has_tools:
                    st.write("(No response)")

# 7. Security Action Handling (Pending Streamlit Permission Prompt)
if st.session_state.get("pending_action"):
    pending = st.session_state["pending_action"]
    p_type = pending.get("type", "action")
    p_target = pending.get("target", "")

    action_label = "Terminal Execution" if p_type == "terminal" else "Filesystem Access Outside Sandbox"
    prompt_desc = (
        f"The assistant is requesting permission to execute shell command:"
        if p_type == "terminal"
        else f"The assistant is requesting read/write access to path outside workspace:"
    )

    st.markdown(
        f"""
        <div class="security-card">
            <div class="security-header">🛡️ Security Authorization Required</div>
            <div class="security-body">
                <strong>{action_label}</strong>: {prompt_desc}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.code(p_target, language="bash" if p_type == "terminal" else "text")

    col_y, col_n = st.columns([1, 1])
    with col_y:
        if st.button("✅ Allow (Y)", type="primary", use_container_width=True, key="btn_allow_action"):
            act = st.session_state.pop("pending_action")

            if act["type"] == "filesystem":
                target_dir = Path(act.get("target_dir") or Path(os.path.expanduser(act["target"])).resolve())
                if not target_dir.is_dir():
                    target_dir = target_dir.parent
                session_approved_dirs.add(target_dir)

            func_resp = run_tool(act["tool_call"])
            content_str = func_resp if isinstance(func_resp, str) else json.dumps(func_resp)
            tool_msg = {
                "role": "tool",
                "tool_call_id": act["call_id"],
                "content": content_str,
            }
            st.session_state["messages"].append(tool_msg)
            if current_conv_id:
                save_message(current_conv_id, tool_msg)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Executing approved action and continuing..."):
                    res = execute_agent_turn(
                        messages=st.session_state["messages"],
                        conversation_id=current_conv_id,
                        client_instance=client,
                        model_name=MODEL,
                        approved_dirs=session_approved_dirs,
                        epochs=EPOCHS,
                        max_request_tokens=MAX_REQUEST_TOKENS,
                        completion_reservation=MAX_COMPLETION_TOKENS,
                        interactive=False,
                    )
                    st.session_state["messages"] = res.messages
                    if getattr(res, "pending_action", None):
                        st.session_state["pending_action"] = res.pending_action

            st.rerun()

    with col_n:
        if st.button("❌ Deny (N)", use_container_width=True, key="btn_deny_action"):
            act = st.session_state.pop("pending_action")

            if act["type"] == "filesystem":
                denial_msg = f"Access denied by user to path: {act['target']}. You must ask the user for an alternative path."
            else:
                denial_msg = "Command execution denied by user. Do not attempt to run this command without permission."

            tool_msg = {
                "role": "tool",
                "tool_call_id": act["call_id"],
                "content": json.dumps(denial_msg),
            }
            st.session_state["messages"].append(tool_msg)
            if current_conv_id:
                save_message(current_conv_id, tool_msg)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Processing denial and continuing..."):
                    res = execute_agent_turn(
                        messages=st.session_state["messages"],
                        conversation_id=current_conv_id,
                        client_instance=client,
                        model_name=MODEL,
                        approved_dirs=session_approved_dirs,
                        epochs=EPOCHS,
                        max_request_tokens=MAX_REQUEST_TOKENS,
                        completion_reservation=MAX_COMPLETION_TOKENS,
                        interactive=False,
                    )
                    st.session_state["messages"] = res.messages
                    if getattr(res, "pending_action", None):
                        st.session_state["pending_action"] = res.pending_action

            st.rerun()

# 8. User Chat Input
user_prompt = st.chat_input("Message Local AI Assistant...")

if user_prompt:
    # 8.1 Append & Save User Message
    user_msg_dict = {"role": "user", "content": user_prompt}
    st.session_state["messages"].append(user_msg_dict)
    if current_conv_id:
        save_message(current_conv_id, user_msg_dict)

    escaped_user = html.escape(user_prompt)
    st.markdown(
        f"""
        <div class="chat-row-user">
            <div class="user-bubble">
                <div class="user-bubble-header">
                    <span>You</span> <span>👤</span>
                </div>
                <div class="user-bubble-content">{escaped_user}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 8.2 Execute Multi-Step Turn
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking and executing tools..."):
            res = execute_agent_turn(
                messages=st.session_state["messages"],
                conversation_id=current_conv_id,
                client_instance=client,
                model_name=MODEL,
                approved_dirs=session_approved_dirs,
                epochs=EPOCHS,
                max_request_tokens=MAX_REQUEST_TOKENS,
                completion_reservation=MAX_COMPLETION_TOKENS,
                interactive=False,
            )

        st.session_state["messages"] = res.messages
        if getattr(res, "pending_action", None):
            st.session_state["pending_action"] = res.pending_action

    st.rerun()
