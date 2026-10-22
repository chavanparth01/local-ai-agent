from datetime import datetime
from dotenv import load_dotenv
import requests
import os
from Utils.utilities import log, get_api_json
import json

# *Global Variables
user_city = ""

load_dotenv()


# !Tool1 -> Current Time Tool
def getCurrentTime() : 
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# !Tool2 -> Get Weather Tool
def get_weather(city):

    # ?Get current weather data using this free weather API 
    base_url = "http://api.weatherapi.com/v1/current.json"
    api_key = os.getenv("WEATHER_API_KEY")

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
def get_user_city() : 
    if user_city == "":
        user_city = input("Enter your city: ")

    return user_city

# !Tool4 -> For the weather forecast
def get_forecast(city, days, start_day=0):
    # Validate the forecast range
    if days < 1 or start_day < 0 or start_day + days > 14:
        raise ValueError(
            "Weather forecast is available for a maximum of 14 days. "
            "Please request a valid forecast range."
        )

    base_url = "https://api.weatherapi.com/v1/forecast.json"
    api_key = os.getenv("WEATHER_API_KEY")

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

    for item in data["results"]:
        results.append({
            "title": item["title"],
            "content" : item["content"][:500],
            "url" : item["url"],
            "score" : item["score"],
        })


    return results

# !Function to run tools
def run_tool(tool_call):
    func_name = tool_call.function.name
    func_arguments = tool_call.function.arguments

    actual_func = function_map[func_name]
    args = json.loads(func_arguments)

    try:
        func_response = actual_func(**args)
        log(f"Tool {func_name} called successfully with arguments : {args} and response : {func_response}")

    except Exception as e:
        func_response = f"error while running function {func_name} : {str(e)}"

    
    return func_response


# !function map 
function_map = {
    "getCurrentTime" : getCurrentTime,
    "get_weather" : get_weather,
    "get_user_city" : get_user_city,
    "get_forecast" : get_forecast,
    "web_search" : web_search,
}
