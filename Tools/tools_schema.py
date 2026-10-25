
# !SCHEMA FOR ALL THE TOOLS USED
GET_CURRENT_TIME = {
    "type": "function",
    "function": {
        "name": "getCurrentTime",
        "description": "Return the current date and time in YYYY-MM-DD HH:MM:SS format.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

GET_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Return the current weather data for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city for which current weather is requested."
                }
            },
            "required": ["city"]
        }
    }
}

GET_USER_CITY = {
    "type": "function",
    "function": {
        "name": "get_user_city",
        "description": (
            "Retrieve the user's current city from available location "
            "information. Use when the user asks about their location or "
            "needs a city for a location-dependent question. Never guess "
            "if location information is unavailable."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

GET_FORECAST = {
    "type": "function",
    "function": {
        "name": "get_forecast",
        "description": (
            "Return a city's weather forecast for the requested number "
            "of days, including maximum and minimum temperatures, conditions, "
            "and chance of rain. Use for future weather, not current weather."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city for which the forecast is requested."
                },
                "days": {
                    "type": "integer",
                    "description": (
                        "Number of forecast days, from 1 to 14. "
                        "Choose the number based on the user's request."
                    )
                },
                "start_day": {
                "type": "integer",
                "description": (
                    "Number of days from today to begin the forecast. "
                    "Use 0 for today, 1 for tomorrow, and 3 for three days from today and so on."
                )
            }
            },
            "required": ["city", "days", "start_day"]
        }
    }
}

WEB_SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for information, current events, facts, or topics "
            "requiring external research. Use the results to formulate an "
            "answer and include source URLs when relevant."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A concise, specific web search query."
                    )
                }
            },
            "required": ["query"]
        }
    }
}

USER_FACTS = {
        "type": "function",
        "function": {
            "name": "save_user_facts_to_db",
            "description": "Saves a permanent fact about the user to long-term memory. Use this whenever the user reveals their name, location, technical preferences, or other persistent details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "A clear, concise statement of the fact to remember, e.g., 'User prefers C++ for competitive programming'."
                    }
                },
                "required": ["fact"],
            },
        }
    }

DELETE_USER_FACT = {
        "type": "function",
        "function": {
            "name": "delete_user_fact",
            "description": "Deletes an outdated or incorrect user fact from long-term memory using its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact_id": {
                        "type": "integer",
                        "description": "The integer ID of the fact to delete, as provided in the system prompt."
                    }
                },
                "required": ["fact_id"],
            },
        }
    }

UPDATE_USER_FACT = {
        "type": "function",
        "function": {
            "name": "update_user_fact",
            "description": "Updates an existing user fact in long-term memory using its ID. Use this when a user changes a previously stated preference (e.g., changes their PIN or moves to a new city).",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact_id": {
                        "type": "integer",
                        "description": "The integer ID of the fact to update."
                    },
                    "new_fact": {
                        "type": "string",
                        "description": "The new, updated fact string."
                    }
                },
                "required": ["fact_id", "new_fact"],
            },
        }
    }

READ_FILE = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a local file. Supports standard tilde shortcuts like ~/Desktop.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to read."
                }
            },
            "required": ["file_path"]
        }
    }
}

WRITE_FILE = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write content to a local file. Automatically creates any missing parent directories.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to write."
                },
                "content": {
                    "type": "string",
                    "description": "The text content to write into the file."
                }
            },
            "required": ["file_path", "content"]
        }
    }
}

LIST_DIRECTORY = {
    "type": "function",
    "function": {
        "name": "list_directory",
        "description": "List the contents (files and subdirectories) of a local directory. Supports standard tilde shortcuts like ~/Desktop.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory_path": {
                    "type": "string",
                    "description": "The path to the directory to list."
                }
            },
            "required": ["directory_path"]
        }
    }
}

EXECUTE_COMMAND = {
    "type": "function",
    "function": {
        "name": "execute_command",
        "description": "Executes a shell command in the terminal. Use this to run scripts, compile code, list processes, or check system configurations. Ensure the commands are appropriate for the user's current operating system (Windows, macOS, or Linux).",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The exact shell command to execute."
                }
            },
            "required": ["command"],
        },
    }
}

# Combine all tool schemas
Tools = [
    GET_CURRENT_TIME,
    GET_WEATHER,
    GET_USER_CITY,
    GET_FORECAST,
    WEB_SEARCH,
    USER_FACTS,
    DELETE_USER_FACT,
    UPDATE_USER_FACT,
    READ_FILE,
    WRITE_FILE,
    LIST_DIRECTORY,
    EXECUTE_COMMAND,
]
