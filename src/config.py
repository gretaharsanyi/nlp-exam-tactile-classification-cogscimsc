import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

# Models
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")

# Length constraints (you can change MAX_CHARS here anytime)
MIN_CHARS = 40
MAX_CHARS = 110

# Metaphor markers (used for literal=0 vs literal=1 rule)
METAPHOR_MARKERS = [" like ", " as if ", " as though "]

# Outputs
OUT_SAMPLES = "data/generated/raw/samples.jsonl"
OUT_REJECTS = "data/generated/raw/rejects.jsonl"
JOB_LOG = "logs/job_log.csv"

# If True: clear samples/rejects/joblog at the start of each run (OVERWRITE)
RESET_OUTPUTS_EACH_RUN = False

# Less “book prose”, more description-like
STYLE_SEEDS = [
    "plain and direct",
    "concise",
    "matter-of-fact",
    "simple and descriptive",
    "neutral tone",
    "slightly vivid but not poetic",
]
