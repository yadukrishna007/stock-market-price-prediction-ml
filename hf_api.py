import os
import logging
from typing import Any, Dict

import requests


logger = logging.getLogger(__name__)

API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ.get('HF_TOKEN', 'use ur hfugging face token')}",
}


def query(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not headers["Authorization"].strip():
        raise RuntimeError("HF_TOKEN environment variable is not set.")

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        logger.exception("Timeout while calling HuggingFace Router API.")
        raise
    except requests.exceptions.RequestException as exc:
        logger.exception("HTTP error while calling HuggingFace Router API: %s", exc)
        raise


def generate_ai_response(ticket_text: str) -> str:
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a professional customer support assistant.",
            },
            {"role": "user", "content": ticket_text},
        ],
        "model": "deepseek-ai/DeepSeek-R1:novita",
    }
    try:
        result = query(payload)
        return result["choices"][0]["message"]["content"]
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("LLM error: %s", exc)
        return f"LLM Error: {str(exc)}"


__all__ = ["generate_ai_response"]
