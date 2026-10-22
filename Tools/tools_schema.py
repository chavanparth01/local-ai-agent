
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

# Combine all tool schemas
Tools = [
    GET_CURRENT_TIME,
    GET_WEATHER,
    GET_USER_CITY,
    GET_FORECAST,
    WEB_SEARCH
]
