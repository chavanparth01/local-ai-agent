# *standard imports : 
import os
import time

# *Third party imports : 
import requests

# ANSI Color Codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
DIM = "\033[2m"
RESET = "\033[0m"


def is_debug_enabled():
    """Check if debug logging is enabled via environment variable."""
    return (os.getenv("DEBUG") or "").lower() in ["true", "1", "yes"]


def log(message):
    """General debug logger."""
    if is_debug_enabled():
        print(f"    {RED}{message}{RESET}")


def get_provider_info(base_url=None, model=None):
    """
    Detect the active LLM provider based on BASE_URL and model name.
    Supports seamless switching between Groq Cloud, Local Ollama, and OpenAI.
    """
    base_url = (base_url or os.getenv("BASE_URL") or "").strip().lower()
    model = (model or os.getenv("MODEL") or "").strip()

    if "googleapis.com" in base_url or "gemini" in model.lower():
        provider = "Google Gemini"
        is_cloud = True
    elif "groq.com" in base_url or "openai/gpt-oss-120b" in model:
        provider = "Groq Cloud"
        is_cloud = True
    elif "localhost" in base_url or "127.0.0.1" in base_url or "11434" in base_url:
        provider = "Local Ollama"
        is_cloud = False
    elif "api.openai.com" in base_url:
        provider = "OpenAI Direct"
        is_cloud = True
    else:
        provider = "OpenAI-Compatible Endpoint"
        is_cloud = not ("localhost" in base_url or "127.0.0.1" in base_url)

    return {
        "provider": provider,
        "is_cloud": is_cloud,
        "base_url": base_url or "default",
        "model": model or "unknown"
    }


def log_llm_request(model, duration_s, finish_reason=None, tool_names=None, usage=None):
    """
    Log diagnostic information about an LLM completion request.
    Never logs raw prompts or sensitive keys.
    """
    if not is_debug_enabled():
        return

    info = get_provider_info(model=model)
    provider_tag = f"{CYAN}[{info['provider']}: {model}]{RESET}"
    latency_tag = f"{YELLOW}{duration_s:.3f}s{RESET}"
    reason_tag = f"{DIM}finish_reason={finish_reason}{RESET}" if finish_reason else ""
    
    tools_str = ""
    if tool_names:
        tools_str = f" | {MAGENTA}tools_requested={tool_names}{RESET}"

    usage_str = ""
    if usage:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if prompt_tokens is not None and completion_tokens is not None:
            usage_str = f" | {DIM}tokens={prompt_tokens}+{completion_tokens}{RESET}"

    print(f"  ⚡ {provider_tag} Response in {latency_tag} {reason_tag}{tools_str}{usage_str}")


def log_tool_execution(tool_name, duration_s, success=True, error=None):
    """
    Log diagnostic timing and status for a local tool call.
    """
    if not is_debug_enabled():
        return

    status_tag = f"{GREEN}SUCCESS{RESET}" if success else f"{RED}FAILED{RESET}"
    timing = f"{YELLOW}{duration_s:.4f}s{RESET}"
    err_str = f" | error: {error}" if error else ""
    print(f"  🔧 {DIM}Tool:{RESET} {CYAN}{tool_name}{RESET} -> {status_tag} in {timing}{err_str}")


def get_api_json(url, headers, payload, timeout):
    """HTTP helper for external REST APIs."""
    try:
        log("Starting API request")

        if not headers:
            response = requests.get(
                url,
                params=payload,
                timeout=timeout
            )
        else:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout
            )

        log(f"HTTP status: {response.status_code}")

        response.raise_for_status()

        data = response.json()

        log("API request completed and JSON parsed successfully")
        return data

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "The external API timed out. Please try again."
        ) from exc

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Could not connect to the external API. "
            "Check your network connection."
        ) from exc

    except requests.exceptions.HTTPError as exc:
        status = (
            exc.response.status_code
            if exc.response is not None
            else "unknown"
        )
        raise RuntimeError(
            f"The external API returned HTTP {status}."
        ) from exc

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            "An unexpected network error occurred."
        ) from exc

    except ValueError as exc:
        raise RuntimeError(
            "The external API returned invalid JSON."
        ) from exc


def get_api_key(API):
    """Retrieve an API key from the environment."""
    key = os.getenv(API)

    if not key:
        raise Exception(f"The api key {API} is not configured in the env file")

    return key


# =====================================================================
# Token-Aware Context Management & Estimation Utilities
# =====================================================================
import math
import json


def estimate_tokens(text: str) -> int:
    """
    Conservative token estimation for text and JSON content.
    Assumes ~3.2 characters per token to ensure our estimation remains
    safely above actual BPE counts without underestimating.
    """
    if not text:
        return 0
    return math.ceil(len(str(text)) / 3.2)


def estimate_message_tokens(message: dict) -> int:
    """
    Estimates the token count of a single message dictionary,
    including ChatML framing overhead (<|im_start|>{role}\n{content}<|im_end|>).
    """
    if not isinstance(message, dict):
        return 0

    # Base ChatML framing overhead per message (4 tokens)
    tokens = 4

    content = message.get("content")
    if content:
        tokens += estimate_tokens(str(content))

    # Assistant tool calls definition overhead
    tool_calls = message.get("tool_calls")
    if tool_calls and isinstance(tool_calls, list):
        for tc in tool_calls:
            tokens += 6  # Framing for tool call declaration
            if isinstance(tc, dict):
                func = tc.get("function") or {}
                name = func.get("name") or ""
                args = func.get("arguments") or ""
                tokens += estimate_tokens(name) + estimate_tokens(args)

    # Tool response role message
    if message.get("role") == "tool":
        t_id = message.get("tool_call_id") or ""
        tokens += estimate_tokens(t_id) + 2

    return tokens


def estimate_tools_tokens(tools: list) -> int:
    """
    Estimates token footprint of the tools schema list passed in the request.
    """
    if not tools:
        return 0
    try:
        serialized = json.dumps(tools)
        return math.ceil(len(serialized) / 3.0) + 16
    except Exception:
        return 1600


def estimate_request_tokens(messages: list, tools: list = None, completion_reservation: int = 0) -> int:
    """
    Calculates the total estimated token footprint of an API completion request:
    messages + tool schemas + reserved completion tokens.
    """
    msg_tokens = sum(estimate_message_tokens(m) for m in (messages or []))
    tool_tokens = estimate_tools_tokens(tools) if tools else 0
    return msg_tokens + tool_tokens + completion_reservation


def get_provider_budgets(provider_info: dict, max_request_tokens_env=None, max_completion_tokens_env=None):
    """
    Determines token budget limits based on the active provider and model.
    - Google Gemini (gemini-3.5-flash-lite):
      Dashboard quotas: 15 RPM, 250,000 TPM, 500 RPD.
      Allocates 64,000 request tokens with 4,096 completion tokens.
    - Groq Cloud:
      30,000 TPM limit on 8B models (12,000 request tokens).
      8,000 TPM limit on 120B models (6,000 request tokens).
    - Local / Ollama:
      16,000 request tokens.
    """
    base_url = str(provider_info.get("base_url", "")).lower()
    model = str(provider_info.get("model", "")).lower()
    provider = str(provider_info.get("provider", "")).lower()

    is_gemini = "gemini" in model or "googleapis.com" in base_url or "gemini" in provider
    is_groq = "groq" in provider or "groq" in base_url

    # Allow environment overrides if specified
    env_req = max_request_tokens_env or os.getenv("MAX_REQUEST_TOKENS")
    env_comp = max_completion_tokens_env or os.getenv("MAX_COMPLETION_TOKENS")

    if is_gemini:
        default_req = 64000
        default_comp = 4096
    elif is_groq:
        if "llama-3.1-8b" in model or "llama3-8b" in model or "8b" in model:
            default_req = 12000
            default_comp = 2048
        else:
            default_req = 6000
            default_comp = 1500
    else:
        default_req = 16000
        default_comp = 2048

    if env_req:
        try:
            max_request = int(env_req)
        except ValueError:
            max_request = default_req
    else:
        max_request = default_req

    if env_comp:
        try:
            max_comp = int(env_comp)
        except ValueError:
            max_comp = default_comp
    else:
        max_comp = default_comp

    return max_request, max_comp


def log_context_budget(estimated_tokens, budget_limit, kept_count, total_count, tools_tokens=0, completion_tokens=0):
    """
    Logs diagnostic token budget allocation without exposing conversation contents.
    """
    if not is_debug_enabled():
        return

    margin = budget_limit - estimated_tokens
    print(
        f"  📊 {DIM}Context Budget:{RESET} Estimated {CYAN}{estimated_tokens}{RESET} tokens "
        f"/ Budget {budget_limit} ({YELLOW}{margin} margin{RESET}) | "
        f"Retained {kept_count}/{total_count} msgs | "
        f"Tools: {tools_tokens}t, Completion: {completion_tokens}t"
    )


def is_oversized_request_error(exc) -> bool:
    """
    Distinguishes an HTTP 413 or token-size error from an ordinary rate-limit (RPM) error.
    """
    status_code = getattr(exc, "status_code", None)
    err_body = getattr(exc, "body", None) or {}
    err_msg = str(getattr(exc, "message", exc)).lower()

    if status_code == 413:
        return True
    if "request too large" in err_msg or "reduce your message size" in err_msg:
        return True
    if "tpm" in err_msg and "limit" in err_msg:
        return True

    if isinstance(err_body, dict):
        err_detail = err_body.get("error", {})
        if isinstance(err_detail, dict):
            if err_detail.get("code") == "rate_limit_exceeded" and err_detail.get("type") == "tokens":
                return True
            inner_msg = str(err_detail.get("message", "")).lower()
            if "request too large" in inner_msg or "reduce your message size" in inner_msg:
                return True

    return False

