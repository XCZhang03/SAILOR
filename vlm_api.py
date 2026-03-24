import os
os.environ['OPENAI_API_KEY'] = "sk-qdeVeTHVKNETsdY0ZJtEHQqHIUAhk32IL6reMv1x3AG5lukI"
os.environ['OPENAI_BASE_URL'] = "http://14.103.68.46/v1"
from typing import List
from google import genai
from google.genai import types
from openai import OpenAI
import base64
import time

def build_gemini_input(content: List):
    """Build Gemini input content from a list of (text, image) tuples."""
    gemini_content = []
    for item in content:
        if isinstance(item, str):
            gemini_content.append(item)
        elif isinstance(item, bytes):
            gemini_content.append(types.Part.from_bytes(
                            data=item,
                            mime_type='image/jpeg',
                        ),)
        else:
            raise ValueError("Unsupported content type: {}".format(type(item)))
    return gemini_content

def build_openai_content(content: List):
    """Build OpenAI content from a list of (text, image) tuples."""
    openai_content = []
    for item in content:
        if isinstance(item, str):
            openai_content.append({"type": "input_text", "text": item})
        elif isinstance(item, bytes):
            openai_content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64.b64encode(item).decode('utf-8')}"})
        else:
            raise ValueError("Unsupported content type: {}".format(type(item)))
    return openai_content




# Gemini Models
# MODEL_ID = "gemini-robotics-er-1.5-preview"
# MODEL_ID = "gemini-3-flash-preview"
# MODEL_ID = "gemini-2.5-flash"
# The client gets the API key from the environment variable `GEMINI_API_KEY`.
# client = genai.Client(api_key="jzmwbIL1QJG1vT2AC8yTZzJJFA80xeBGVtrTxfkTHmGu5GIE", http_options=types.HttpOptions(base_url="https://go.apis.huit.harvard.edu/ais-google-gemini/"))



# OpenAI Models
MODEL_ID = "gpt-5.4"
client = OpenAI()



def call_api_gemini(content, thinking=None):
    content = build_gemini_input(content)
    config = None
    if thinking == "low":
        if MODEL_ID == "gemini-2.5-flash":
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=256)
                # Turn off thinking:
                # thinking_config=types.ThinkingConfig(thinking_budget=0)
                # Turn on dynamic thinking:
                # thinking_config=types.ThinkingConfig(thinking_budget=-1)
            )
        elif MODEL_ID == "gemini-3-flash-preview":
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="low")
        )

    elif thinking == "medium":
        if MODEL_ID == "gemini-2.5-flash":
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=1024)
            )
        elif MODEL_ID == "gemini-3-flash-preview":
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="medium")
            )
    start = time.time()

    max_retries = 2
    # Initial wait time of 10s, doubling each time (10, 20, 40, 80...)
    wait_time = 10

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

def call_api_openai(content, thinking=None):
    content = build_openai_content(content)
    if thinking is not None:
        reasoning = {"effort": thinking}
    else:
        reasoning = None
    start = time.time()
    response = client.responses.create(
        model=MODEL_ID,
        input=[
            {"role": "system", "content": "You are a helpful assistant for controlling a robot arm in a simulated environment. You will give advice on how to complete the task based on the image inputs."},
            {"role": "user", "content": content}
        ],
        reasoning=reasoning
    )
    print("Time taken for OpenAI API call: {:.2f} seconds".format(time.time() - start))
    print("Token used: ", response.usage)
    return response.output_text

def call_api(content, thinking=None):
    if MODEL_ID.startswith("gemini"):
        return call_api_gemini(content, thinking)
    elif MODEL_ID.startswith("gpt"):
        return call_api_openai(content, thinking)
    else:
        raise ValueError(f"Unsupported MODEL_ID: {MODEL_ID}")

if __name__ == "__main__":
    content = ['Hi, who are you']
    response = call_api(content, thinking="low")
    print("Response:", response)