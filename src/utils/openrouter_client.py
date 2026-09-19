"""Thin wrapper around the OpenAI SDK, pointed at OpenRouter instead of OpenAI."""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

RETRY_DELAY_SECONDS = 2


def call_model(model_slug: str, messages: list[dict], temperature: float = 0.7) -> str:
    """Send a chat completion request to `model_slug` via OpenRouter and return the reply text."""
    last_error = None
    for attempt in range(2):
        try:
            response = _client.chat.completions.create(
                model=model_slug,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            if attempt == 0:
                time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"OpenRouter call failed for model '{model_slug}' after 2 attempts: {last_error}"
    ) from last_error


if __name__ == "__main__":
    model_a = os.getenv("MODEL_A")
    model_b = os.getenv("MODEL_B")
    test_messages = [{"role": "user", "content": "Say hello in one sentence."}]

    for label, model_slug in (("MODEL_A", model_a), ("MODEL_B", model_b)):
        if not model_slug:
            print(f"{label}: not set in .env, skipping")
            continue
        try:
            reply = call_model(model_slug, test_messages)
            print(f"{label} ({model_slug}): {reply}")
        except Exception as e:
            print(f"{label} ({model_slug}) FAILED: {e}")
