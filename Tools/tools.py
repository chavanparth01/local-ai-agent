# *standarad imports : 
from datetime import datetime
import os
import sys
import pathlib
import json
import subprocess
import time

# *Third party imports : 
from dotenv import load_dotenv

# *module imports : 
from Storage.storage import (
    save_user_fact,
    update_user_fact_db,
    delete_user_fact_db,
    get_user_facts,
)
from Utils.utilities import log, get_api_json, log_tool_execution

# User City Helpers
def get_saved_user_city():
    """Retrieve the saved city from the SQLite user_facts table if it exists."""
    try:
        facts = get_user_facts()
        for _, fact in facts:
            lower = fact.lower()
            if "user city:" in lower or "city:" in lower:
                return fact.split(":", 1)[1].strip()
            if "city is:" in lower or "city is" in lower:
                return fact.split("is", 1)[1].replace(":", "").strip()
            if "departure city is:" in lower:
                return fact.split("is", 1)[1].replace(":", "").strip()
            if "lives in" in lower:
                return fact.split("lives in", 1)[1].replace(":", "").strip()
    except Exception:
        pass
    return None


def save_user_city(city: str):
    """Save or update the user city in the SQLite user_facts table."""
    cleaned = (city or "").strip()
    if not cleaned:
        return
    try:
        facts = get_user_facts()
        for fid, ftext in facts:
            lower = ftext.lower()
            if any(k in lower for k in ["user city:", "city:", "city is", "departure city is"]):
                update_user_fact_db(fid, f"User city: {cleaned}")
                return
        save_user_fact(f"User city: {cleaned}")
    except Exception:
        pass

# Backward compatibility alias
set_user_city = save_user_city

load_dotenv()

# !Tool1 -> Current Time Tool
def getCurrentTime() : 
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# !Tool2 -> Get Weather Tool
def get_weather(city):

    # ?Get current weather data using this free weather API 
    base_url = "https://api.weatherapi.com/v1/current.json"
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        raise ValueError("WEATHER_API_KEY is not configured in the environment.")

    params = {
        "key" : api_key,
        "q" : city,
        "aqi" : "no"
    }

    data = get_api_json(base_url, {}, params, 10)

    return {
        "city" : data["location"]["name"],
        "region" : data["location"]["region"],
        "country" : data["location"]["country"],
        "temperature_c" : data["current"]["temp_c"],
        "condition" : data["current"]["condition"]["text"],
        "wind_kph" : data["current"]["wind_kph"],
        "humidity" : data["current"]["humidity"],
        "feelslike_c" : data["current"]["feelslike_c"]
    }

# !Tool3 -> Get the user's city
def get_user_city(): 
    """
    Returns the user's city.
    1. First checks the SQLite user_facts table. If already saved, reuses it.
    2. Otherwise, asks the user through the interface, saves it to user_facts, and reuses it.
    """
    saved_city = get_saved_user_city()
    if saved_city:
        return saved_city

    # Web mode (Streamlit)
    if os.getenv("ASSISTANT_MODE") == "web":
        return "City is not configured yet. Please enter your city in the sidebar."

    # CLI mode
    try:
        entered = input("Please enter your city: ").strip()
        if entered:
            save_user_city(entered)
            return entered
        return "Unknown city"
    except (EOFError, KeyboardInterrupt):
        return "Unknown city"

# !Tool4 -> For the weather forecast
def get_forecast(city, days, start_day=0):
    days = int(days)
    start_day = int(start_day)

    # Validate the forecast range
    if days < 1 or start_day < 0 or start_day + days > 14:
        raise ValueError(
            "Weather forecast is available for a maximum of 14 days. "
            "Please request a valid forecast range."
        )

    base_url = "https://api.weatherapi.com/v1/forecast.json"
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        raise ValueError("WEATHER_API_KEY is not configured in the environment.")

    params = {
        "key": api_key,
        "q": city,
        "days": days + start_day,  # Changed
        "aqi": "no",
    }

    data = get_api_json(base_url, {}, params, 10)
    forecasts = []

    for day in data["forecast"]["forecastday"][start_day:]:
        forecasts.append({
            "city": data["location"]["name"],
            "date": day["date"],
            "max_temp_c": day["day"]["maxtemp_c"],
            "min_temp_c": day["day"]["mintemp_c"],
            "condition": day["day"]["condition"]["text"],
            "chance_of_rain": day["day"]["daily_chance_of_rain"]
        })

        if len(forecasts) == days:
            break

    return forecasts


# !Tool5 -> Adding the most important web_search tool 
def web_search(query):
    base_url = "https://api.tavily.com/search"
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not configured in the environment.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "query" : query,
        "search_depth" : "basic",
        "max_results" : 5,
    }

    data = get_api_json(base_url, headers, payload, 20)

    results = []

    for item in data.get("results", []):
        results.append({
            "title": item.get("title", ""),
            "content" : (item.get("content") or "")[:500],
            "url" : item.get("url", ""),
            "score" : item.get("score", 0),
        })


    return results


# !Tool6 -> Save User Facts for Long Term Memory
def save_user_facts_to_db(fact):
    try:
        save_user_fact(fact)
        return f"Successfully saved fact to memory: {fact}"

    except Exception as e:
        return f"Failed to save fact: {str(e)}"

# !Tool7 -> Delete User Fact
def delete_user_fact(fact_id):
    try:
        delete_user_fact_db(int(fact_id))
        return f"Successfully deleted fact ID: {fact_id}"
    except Exception as e:
        return f"Failed to delete fact: {str(e)}"

# !Tool8 -> Update User Fact
def update_user_fact(fact_id, new_fact):
    try:
        update_user_fact_db(int(fact_id), new_fact)
        return f"Successfully updated fact ID {fact_id} to: {new_fact}"
    except Exception as e:
        return f"Failed to update fact: {str(e)}"


# !Tool9 -> Read Local File
def read_file(file_path):
    try:
        path = pathlib.Path(os.path.expanduser(file_path)).resolve()
        if not path.is_file():
            return f"Error: File not found at '{file_path}'."
        content = path.read_text(encoding="utf-8", errors="replace")
        max_chars = 3500
        if len(content) > max_chars:
            omitted = len(content) - 3000
            return (
                content[:2500]
                + f"\n...[Output truncated: {omitted} characters omitted to preserve context budget]...\n"
                + content[-500:]
            )
        return content
    except Exception as e:
        return f"Failed to read file: {str(e)}"

# !Tool10 -> Write Local File
def write_file(file_path, content):
    try:
        path = pathlib.Path(os.path.expanduser(file_path)).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote to '{file_path}'."
    except Exception as e:
        return f"Failed to write file: {str(e)}"

# !Tool11 -> List Local Directory
def list_directory(directory_path):
    try:
        path = pathlib.Path(os.path.expanduser(directory_path)).resolve()
        if not path.is_dir():
            return f"Error: Directory not found at '{directory_path}'."
        entries = []
        for item in sorted(path.iterdir()):
            item_type = "DIR" if item.is_dir() else "FILE"
            entries.append(f"[{item_type}] {item.name}")
        max_entries = 80
        if len(entries) > max_entries:
            omitted = len(entries) - max_entries
            entries = entries[:max_entries] + [f"...[Output truncated: {omitted} additional directory entries omitted to preserve context budget]..."]
        return entries
    except Exception as e:
        return f"Failed to list directory: {str(e)}"


# !Tool12 -> Execute Terminal Command,
def execute_command(command):
    """Executes a shell command and returns the standard output and error."""
    try:
        # timeout=15 ensures the agent can't freeze your system with infinite loops
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15
        )
        
        output = result.stdout.strip()
        error = result.stderr.strip()
        
        max_chars = 3500
        if len(output) > max_chars:
            omitted = len(output) - 3000
            output = (
                output[:2500]
                + f"\n...[Output truncated: {omitted} characters omitted to preserve context budget]...\n"
                + output[-500:]
            )
        if len(error) > 2000:
            omitted = len(error) - 1700
            error = (
                error[:1500]
                + f"\n...[Error output truncated: {omitted} characters omitted]...\n"
                + error[-200:]
            )

        if result.returncode == 0:
            return output if output else "Command executed successfully with no output."
        else:
            return f"Command failed with error code {result.returncode}:\n{error}"
            
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 15 seconds."
    except Exception as e:
        return f"Failed to execute command: {str(e)}"


# !Function to run tools
def run_tool(tool_call):
    start_time = time.time()
    if isinstance(tool_call, dict):
        func_obj = tool_call.get("function") or {}
        func_name = func_obj.get("name", "unknown")
        func_arguments = func_obj.get("arguments", "{}")
    else:
        func_name = getattr(getattr(tool_call, "function", None), "name", "unknown")
        func_arguments = getattr(getattr(tool_call, "function", None), "arguments", "{}")
    try:

        if func_name not in function_map:
            err = f"error: Tool '{func_name}' is not recognized."
            log_tool_execution(func_name, time.time() - start_time, success=False, error=err)
            return err

        actual_func = function_map[func_name]
        args = json.loads(func_arguments) if func_arguments else {}

        if not isinstance(args, dict):
            err = f"error: Arguments for tool '{func_name}' must be a dictionary/object."
            log_tool_execution(func_name, time.time() - start_time, success=False, error=err)
            return err

        func_response = actual_func(**args)
        duration = time.time() - start_time
        log_tool_execution(func_name, duration, success=True)
        log(f"Tool {func_name} called successfully with arguments : {args}")

    except Exception as e:
        duration = time.time() - start_time
        func_name = getattr(getattr(tool_call, "function", None), "name", "unknown")
        func_response = f"error while running function {func_name} : {str(e)}"
        log_tool_execution(func_name, duration, success=False, error=str(e))

    return func_response


# !function map 
function_map = {
    "getCurrentTime" : getCurrentTime,
    "get_weather" : get_weather,
    "get_user_city" : get_user_city,
    "get_forecast" : get_forecast,
    "web_search" : web_search,
    "save_user_facts_to_db" : save_user_facts_to_db,
    "save_user_fact" : save_user_facts_to_db,
    "delete_user_fact": delete_user_fact,
    "update_user_fact": update_user_fact,
    "read_file": read_file,
    "write_file": write_file,
    "list_directory": list_directory,
    "execute_command": execute_command,
}
