# Local AI Assistant with Tool Calling & Cloud Reasoning

A Python-based AI assistant powered by **Google Gemini 3.5 Flash-Lite** (with optional support for **Groq Cloud** or local **Ollama**) to answer questions, execute multi-step tools, manage local files and terminal commands with interactive security, retrieve weather, and search the web.

## High-Level Design (HLD)

The diagram below illustrates the project's architecture and how its components interact.

**[View Project HLD on Excalidraw](https://excalidraw.com/#json=fXWcKovPRlTJhQGEAC8en,Vnnx848SDizYeIqt2_bxIg)**

## Tech Stack

- **Python 3.10+** — Application logic, sandbox security, and tool execution
- **Google Gemini Cloud** — High-speed reasoning with `gemini-3.5-flash-lite` (15 RPM, 250K TPM, 500 RPD)
- **Groq Cloud / Ollama** — Flexible alternative providers via OpenAI-compatible endpoints
- **OpenAI Python SDK** — Communicates with Gemini, Groq, or Ollama using standard OpenAI API conventions
- **SQLite3** — Persistent local database for conversations, message history, and long-term user facts
- **WeatherAPI** — Current weather and multi-day forecasts
- **Tavily Search API** — Live web search and citations
- **Requests & python-dotenv** — API calls and environment configuration

## Features

- **Google Gemini Cloud Reasoning** — Powered by `gemini-3.5-flash-lite` for precision tool calling, reasoning, and multi-step tasks.
- **Provider Switching** — Switch seamlessly between Google Gemini, Groq Cloud, and local Ollama simply by updating environment variables.
- **Autonomous Tool Calling** — Dynamically executes tools across web search, weather, system files, and terminal.
- **Persistent Memory & Storage** — SQLite storage (`assistant.db`) for multi-turn chats and long-term fact memory (`save_user_facts_to_db`, `update_user_fact`, `delete_user_fact`).
- **Interactive Security Sandbox** — Confirms user consent (`y/n`) before accessing files outside the workspace or executing terminal commands.
- **Token-Aware Context Management** — Manages context budgets up to 64,000 tokens while maintaining atomic assistant/tool call sequences and preserving SQLite history intact.
- **Diagnostics & Observability** — Structured latency, finish reason, token budget margin, and tool duration logging when `DEBUG=true`.
- **Subprocess Safety** — 15-second execution timeout prevents runaway terminal commands.

## Project Structure

```text
PY/
├── Storage/
│   ├── __init__.py
│   └── storage.py          # SQLite database schema and persistence CRUD
├── Tools/
│   ├── __init__.py
│   ├── tools.py            # Tool implementations & dispatcher with output truncation guards
│   └── tools_schema.py     # 12 OpenAI-compatible JSON tool schemas
├── Utils/
│   ├── __init__.py
│   └── utilities.py        # Token estimation, provider detection, diagnostics, and HTTP helpers
├── tests/
│   ├── test_agent.py       # Integration test suite covering 11 critical scenarios
│   └── test_context_management.py # Regression test suite covering 12 token-budget scenarios
├── app.py                  # Streamlit Web UI application
├── simpleAiAss.py          # Core orchestrator and interactive CLI
├── requirements.txt        # Python package dependencies
├── .env.example            # Environment template for Groq and Ollama
├── .gitignore              # Protects secrets, virtualenvs, and databases
└── README.md
```

## Prerequisites

- **Python 3.10+** installed on macOS, Linux, or Windows
- A free **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/app/apikey) (or Groq Cloud API key from [Groq Console](https://console.groq.com/keys))
- *(Optional for weather)*: A free API key from [WeatherAPI.com](https://www.weatherapi.com/)
- *(Optional for web search)*: A free API key from [Tavily.com](https://tavily.com/)
- *(Optional for offline local mode)*: [Ollama](https://ollama.com/download)

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/chavanparth01/local-ai-assistant.git
cd local-ai-assistant
```

### 2. Create and activate a virtual environment

On macOS or Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` to configure your credentials:

#### Recommended: Google Gemini Configuration
```env
BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
API_KEY=your_gemini_api_key_here
MODEL=gemini-3.5-flash-lite
MAX_REQUEST_TOKENS=64000
MAX_COMPLETION_TOKENS=4096

WEATHER_API_KEY=your_weatherapi_key
TAVILY_API_KEY=your_tavily_api_key
DEBUG=false
```

#### Optional: Groq Cloud or Local Ollama Configuration
```env
# Groq Cloud
BASE_URL=https://api.groq.com/openai/v1
API_KEY=your_groq_api_key
MODEL=llama-3.1-8b-instant

# Or Local Ollama
# BASE_URL=http://localhost:11434/v1
# API_KEY=ollama
# MODEL=qwen3:4b
```

> **Security Note:** Never commit your `.env` file to Git. Your `.gitignore` is pre-configured to keep API keys and SQLite database files (`assistant.db`) private.

## Running the Assistant

### Option A: Modern Web UI (Streamlit)

Launch the interactive web interface:

```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to chat, inspect live tool outputs, switch between past conversations, and view long-term memory facts.

### Option B: Terminal CLI

```bash
python simpleAiAss.py
```

#### CLI Options:
1. **Start a new conversation**: Prompts for a session title and initializes a fresh chat.
2. **Continue a previous conversation**: Displays saved sessions from SQLite and lets you resume any conversation with its full context.

## Running the Automated Test Suite

The project includes an automated test suite verifying tool calling, error recovery, context budgeting, and sandbox guards:

```bash
python -m unittest tests/test_agent.py
```

All mock tests run locally without consuming API credits. If a real API key is present in `.env`, live connectivity and tool execution checks will also execute automatically.

## Security & Privacy Considerations

- **Data Transmitted to Cloud LLM (Gemini / Groq)**: Only the prompts, conversation history (within the context budget), and tool output strings required for reasoning are transmitted via HTTPS to the provider's API.
- **Local Isolation**: The SQLite database (`assistant.db`), private `.env` keys, and local files outside approved directories are **never** automatically sent to the cloud.
- **Filesystem Permissions**: The assistant runs in a restricted sandbox (`session_approved_dirs`). Attempting to read or write files outside the workspace prompts the user for interactive `(y/n)` approval.
- **Terminal Execution**: Shell commands requested via `execute_command` **always** require explicit user approval before execution and have an automatic 15-second timeout.

## Troubleshooting

| Error | Cause | Resolution |
|---|---|---|
| `Authentication Error (401 / 400)` | Missing or invalid Gemini / Groq API key | Check that `API_KEY` in `.env` is accurate without extra quotes or spaces. For Gemini, verify in [Google AI Studio](https://aistudio.google.com/app/apikey). |
| `Rate Limit Error (429)` | Exceeded free-tier TPM/RPM quota (Gemini: 15 RPM, 250K TPM, 500 RPD) | The client will automatically retry after backoff. Alternatively, wait a moment or switch to local Ollama. |
| `Connection Error` | No internet connection or endpoint down | Verify your internet connection or check provider status. If using Ollama, ensure `ollama serve` is running. |
| `Access denied by user` | File/command was rejected in sandbox | Answer `y` at the prompt if you trust the directory or command, or guide the agent in natural language. |
