from google import genai
from google.genai import types
import json
import re
from PIL import Image
from time import sleep



# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key="jzmwbIL1QJG1vT2AC8yTZzJJFA80xeBGVtrTxfkTHmGu5GIE", http_options=types.HttpOptions(base_url="https://go.apis.huit.harvard.edu/ais-google-gemini/"))

# MODEL_ID = "gemini-robotics-er-1.5-preview"
MODEL_ID = "gemini-3-flash-preview"
# MODEL_ID = "gemini-2.5-flash"

g25_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=256)
        # Turn off thinking:
        # thinking_config=types.ThinkingConfig(thinking_budget=0)
        # Turn on dynamic thinking:
        # thinking_config=types.ThinkingConfig(thinking_budget=-1)
    )
g3_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="low")
)

def call_api(content, thinking=None):
    config = None
    if thinking == "low":
        if MODEL_ID == "gemini-2.5-flash":
            config = g25_config
        elif MODEL_ID == "gemini-3-flash-preview":
            config = g3_config
    elif thinking == "medium":
        if MODEL_ID == "gemini-2.5-flash":
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=1024)
            )
        elif MODEL_ID == "gemini-3-flash-preview":
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="medium")
            )
    import time
    start = time.time()

    max_retries = 5
    # Initial wait time of 10s, doubling each time (10, 20, 40, 80...)
    wait_time = 60

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=content,
                config=config,
            )
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2 # Exponential backoff
            else:
                print("Max retries reached. Server is likely down.")
                raise e # Only crash after all retries fail


    end = time.time()
    print(f"API call took {end - start:.2f} seconds.")
    print("Thinking tokens used:", response.usage_metadata.thoughts_token_count)
    print("output tokens used:", response.usage_metadata.candidates_token_count)
    return response.text
    