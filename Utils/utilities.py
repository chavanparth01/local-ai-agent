import os
import requests

start_code = "\033[91m"
end_code = "\033[0m"

def log(message):
    if os.getenv("DEBUG").lower() == "true":
        print(f"    {start_code}{message}{end_code}")  


import requests

def get_api_json(url, headers, payload, timeout):
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
        # Avoid logging the complete URL because it may contain an API key.
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
    key = os.getenv(API)

    if not key:
        raise Exception(f"The api key {API} is not configured in the env file")

