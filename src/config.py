import os
from pathlib import Path
from dotenv import load_dotenv

# loading .env
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

# models
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")

# length constraints
MIN_CHARS = 40
MAX_CHARS = 110

# metaphor markers
METAPHOR_MARKERS = [" like ", " as if ", " as though "]

# outputs
OUT_SAMPLES = "data/generated/raw/samples.jsonl"
OUT_REJECTS = "data/generated/raw/rejects.jsonl"
JOB_LOG = "logs/job_log.csv"

RESET_OUTPUTS_EACH_RUN = False

# style
STYLE_SEEDS = [
    "plain and direct",
    "concise",
    "matter-of-fact",
    "simple and descriptive",
    "neutral tone",
    "slightly vivid but not poetic",
]
