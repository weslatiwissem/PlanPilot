"""The ONLY place in Agent 1 that talks to Gemini.

Later you can swap generate_structured() for the fine-tuned SLM (outlines + PEFT)
without touching clarify.py / scope.py.
"""
import os
import time
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)
_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai  # imported lazily so tests run without the SDK/key
        from google.genai import types

        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set (copy .env.example to .env).")
        timeout_s = float(os.getenv("GEMINI_TIMEOUT_S", "120"))
        _client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),  # milliseconds
        )
    return _client


def generate_structured(
    prompt: str,
    schema: Type[T],
    system: str = "",
    temperature: Optional[float] = None,
    retries: int = 3,
) -> T:
    """Call Gemini and return a validated instance of `schema`.

    Retries on malformed JSON, dropped connections and 5xx errors (with backoff).
    Does NOT retry 4xx errors (bad key, bad model name, quota): retrying cannot fix those.
    """
    from google.genai import errors, types

    config = types.GenerateContentConfig(
        system_instruction=system or None,
        response_mime_type="application/json",
        response_schema=schema,
        temperature=temperature,
    )
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = _get_client().models.generate_content(
                model=model, contents=prompt, config=config
            )
            if isinstance(resp.parsed, schema):
                return resp.parsed
            return schema.model_validate_json(resp.text)
        except errors.ClientError:
            raise  # 4xx: fix the key / model name / quota instead
        except RuntimeError as e:
            if "GEMINI_API_KEY" in str(e):
                raise
            last_error = e
        except (ValidationError, ValueError) as e:  # malformed JSON
            last_error = e
        except Exception as e:  # dropped connection, timeout, 5xx
            last_error = e
        if attempt < retries:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(
        f"Gemini call failed after {retries + 1} attempts: {type(last_error).__name__}: {last_error}"
    )
