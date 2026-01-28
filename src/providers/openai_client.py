import os
from tenacity import retry, wait_exponential, stop_after_attempt
from openai import OpenAI

from src.config import OPENAI_MODEL

_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(5))
def call_openai(system: str, user: str) -> str:
    resp = _openai_client.responses.create(
        model=OPENAI_MODEL,
        instructions=system,
        input=user,
    )
    return resp.output_text
