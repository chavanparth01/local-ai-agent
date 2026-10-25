# Local AI Assistant with Tool Calling & Cloud Reasoning

A Python-based AI assistant that uses **Groq Cloud** with **`openai/gpt-oss-120b`** (or a local **Ollama** model) to answer questions, execute multi-step tools, manage local files and terminal commands with interactive security, retrieve weather, and search the web.

## High-Level Design (HLD)

The diagram below illustrates the project's architecture and how its components interact.

**[View Project HLD on Excalidraw](https://excalidraw.com/#json=fGIPpdaFOtvitql6z395Z,xwtCvoeFQmAYCfC6RUnb3w)**

## Tech Stack

- **Python 3.10+** — Application logic, sandbox security, and tool execution
- **Groq Cloud** — Ultra-fast cloud inference for open-weights reasoning models (`openai/gpt-oss-120b`)
- **Ollama (Optional Fallback)** — Local LLM provider (`qwen3:4b`) for offline environments
- **OpenAI Python SDK** — Communicates with Groq or Ollama through the standard OpenAI-compatible API
- **SQLite3** — Persistent local database for conversations, message history, and long-term user facts
- **WeatherAPI** — Current weather and multi-day forecasts
- **Tavily Search API** — Live web search and citations
- **Requests & python-dotenv** — API calls and environment configuration

## Features

- **Groq Cloud Reasoning** — Powered by `openai/gpt-oss-120b` for precision tool calling, reasoning, and multi-step tasks.
- **Provider Switching** — Switch seamlessly between Groq Cloud and local Ollama simply by updating environment variables.
- **Autonomous Tool Calling** — Dynamically executes tools across web search, weather, system files, and terminal.
- **Persistent Memory & Storage** — SQLite storage (`assistant.db`) for multi-turn chats and long-term fact memory (`save_user_facts_to_db`, `update_user_fact`, `delete_user_fact`).
- **Interactive Security Sandbox** — Confirms user consent (`y/n`) before accessing files outside the workspace or executing terminal commands.
- **Token-Aware Context Management** — Conservative token estimation accounting for system prompts, tools schema, completion reservations, and conversation turns. Prevents Groq Free Tier 8,000 TPM limit breaches while maintaining atomic assistant/tool call sequences and preserving SQLite history intact.
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
├── simpleAiAss.py          # Core orchestrator and interactive CLI
├── requirements.txt        # Python package dependencies
├── .env.example            # Environment template for Groq and Ollama
├── .gitignore              # Protects secrets, virtualenvs, and databases
└── README.md
```

## Prerequisites

- **Python 3.10+** installed on macOS, Linux, or Windows
- A free **Groq Cloud API Key** from [Groq Console](https://console.groq.com/keys)
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

#### Recommended: Groq Cloud Configuration
```env
BASE_URL=https://api.groq.com/openai/v1
API_KEY=gsk_your_groq_api_key_here
MODEL=openai/gpt-oss-120b
CONTEXT_BUDGET_CHARS=32000

WEATHER_API_KEY=your_weatherapi_key
TAVILY_API_KEY=your_tavily_api_key
DEBUG=false
```

#### Optional: Local Ollama Configuration
To run completely offline with local Ollama, update `.env`:
```env
BASE_URL=http://localhost:11434/v1
API_KEY=ollama
MODEL=qwen3:4b
CONTEXT_BUDGET_CHARS=16000
```

> **Security Note:** Never commit your `.env` file to Git. Your `.gitignore` is pre-configured to keep API keys and SQLite database files (`assistant.db`) private.

## Running the Assistant

Activate your virtual environment and run:

```bash
python simpleAiAss.py
```

### CLI Options:
1. **Start a new conversation**: Prompts for a session title and initializes a fresh chat.
2. **Continue a previous conversation**: Displays saved sessions from SQLite and lets you resume any conversation with its full context.

## Running the Automated Test Suite

The project includes an automated test suite verifying tool calling, error recovery, context budgeting, and sandbox guards:

```bash
python -m unittest tests/test_agent.py
```

All mock tests run locally without consuming API credits. If a real `gsk_...` key is present in `.env`, a live connectivity check will also execute automatically.

## Security & Privacy Considerations

- **Data Transmitted to Groq Cloud**: Only the prompts, conversation history (within the context budget), and tool output strings required for reasoning are transmitted via HTTPS to Groq's API.
- **Local Isolation**: The SQLite database (`assistant.db`), private `.env` keys, and local files outside approved directories are **never** automatically sent to the cloud.
- **Filesystem Permissions**: The assistant runs in a restricted sandbox (`session_approved_dirs`). Attempting to read or write files outside the workspace prompts the user for interactive `(y/n)` approval.
- **Terminal Execution**: Shell commands requested via `execute_command` **always** require explicit user approval before execution and have an automatic 15-second timeout.

## Troubleshooting

| Error | Cause | Resolution |
|---|---|---|
| `Authentication Error (401)` | Missing or invalid Groq API key | Check that `API_KEY` in `.env` starts with `gsk_` and has no extra spaces. |
| `Rate Limit Error (429)` | Exceeded free-tier TPM/RPM on Groq | The client will automatically retry after backoff. Alternatively, wait a moment or switch to local Ollama. |
| `Connection Error` | No internet connection or endpoint down | Verify your internet connection or check [status.groq.com](https://status.groq.com). If using Ollama, ensure `ollama serve` is running. |
| `Access denied by user` | File/command was rejected in sandbox | Answer `y` at the prompt if you trust the directory or command, or guide the agent in natural language. |
