import os
from tenacity import retry, wait_exponential, stop_after_attempt
import anthropic

from src.config import CLAUDE_MODEL

_claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(5))
def call_claude(system: str, user: str) -> str:
    msg = _claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=220,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts = []
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()
