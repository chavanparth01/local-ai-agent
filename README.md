# Local AI Assistant with Tool Calling

A Python-based AI assistant that uses a locally running Large Language Model (LLM) to answer questions, call Python tools, retrieve weather information, and search the web when needed.

## High-Level Design (HLD)

The diagram below illustrates the project's architecture and how its components interact.

**[View Project HLD on Excalidraw](https://excalidraw.com/#json=fGIPpdaFOtvitql6z395Z,xwtCvoeFQmAYCfC6RUnb3w)**

## Tech Stack

- **Python** — Application logic and tool execution
- **Ollama** — Runs the language model locally
- **Qwen3 4B** — Language model
- **OpenAI Python SDK** — Communicates with the model through a compatible API
- **WeatherAPI** — Provides current weather and forecast data
- **Web Search** — Retrieves online information
- **Requests** — Makes HTTP requests to external APIs
- **python-dotenv** — Loads configuration from environment variables

## Features

- **Local LLM Execution** — Runs local LLMs via Ollama (or any OpenAI-compatible API).
- **Autonomous Tool Calling** — Agent evaluates queries and dynamically executes Python tools.
- **Persistent Memory & Storage** — SQLite-backed conversation history and long-term user facts memory.
- **Weather & Forecasts** — Retrieves current weather and multi-day forecasts via WeatherAPI.
- **Live Web Search** — Queries real-time web results using Tavily Search API.
- **File System Operations** — Read files, write files, and explore directories on the host machine.
- **Terminal Execution** — Execute shell commands with safety timeouts.
- **Interactive Security Sandbox** — Confirms user consent before accessing paths outside approved directories or executing shell commands.
- **Conversation Session Management** — Pick up previous chat threads or start fresh conversations with context budget limits.

## Project Structure

```text
PY/
├── Storage/
│   ├── __init__.py
│   └── storage.py
├── Tools/
│   ├── __init__.py
│   ├── tools.py
│   └── tools_schema.py
├── Utils/
│   ├── __init__.py
│   └── utilities.py
├── simpleAiAss.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Prerequisites

Before running the project, install:

- Python 3.10 or a compatible version
- [Ollama](https://ollama.com/download)
- A WeatherAPI key from [WeatherAPI.com](https://www.weatherapi.com/) for weather features

## Setup Instructions

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_FOLDER
```

Replace the placeholders with your actual GitHub repository URL and repository folder name.

### 2. Download the model

Download Qwen3 4B using Ollama:

```bash
ollama pull qwen3:4b
```

Make sure Ollama is running before starting the assistant.

### 3. Create a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file in the project root. Use `.env.example` as a reference.

Example configuration:

```env
BASE_URL=http://localhost:11434/v1
API_KEY=ollama
MODEL=qwen3:4b
WEATHER_API_KEY=your_weatherapi_key
TAVILY_API_KEY=your_tavily_api_key
```

Replace `your_weatherapi_key` and `your_tavily_api_key` with your actual API keys.

Keep your actual `.env` file private. Never commit API keys or other secrets to GitHub.

### 6. Run the assistant

Activate your virtual environment if it is not already active, ensure Ollama is running, and execute:

```bash
python simpleAiAss.py
```

You can now interact with the assistant through the terminal.

## How It Works

1. The user enters a prompt.
2. The LLM processes the request and decides whether it can answer directly or needs a tool.
3. If a tool is needed, the application executes the corresponding Python function.
4. The tool result is added to the conversation history.
5. The LLM uses the result to generate a response.
6. For requests requiring online information, the assistant can use its web search functionality, if configured.

## Configuration and Security

- Keep secrets in `.env`.
- Commit `.env.example` with placeholder values only.
- Make sure `.env`, `.venv/`, `__pycache__/`, and log files are excluded through `.gitignore`.
- Never publish API keys or other credentials.

## Future Improvements

- Add more tools and integrations.
- Improve response speed and error handling.
- Add a graphical user interface.
- Expand support for different models and APIs.
