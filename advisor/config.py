"""Settings from the environment, with defaults that need no configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv(ROOT.parent / ".env")


@dataclass(frozen=True)
class Settings:
    api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model: str = os.getenv("ADVISOR_MODEL", "google/gemini-2.5-flash")
    request_timeout: float = float(os.getenv("ADVISOR_LLM_TIMEOUT", "25"))
    max_tokens: int = int(os.getenv("ADVISOR_MAX_TOKENS", "900"))
    memory_dir: Path = Path(os.getenv("ADVISOR_MEMORY_DIR", "/tmp/advisor-memory"))
    max_plan_steps: int = int(os.getenv("ADVISOR_MAX_PLAN_STEPS", "8"))
    max_replans: int = int(os.getenv("ADVISOR_MAX_REPLANS", "2"))

    @property
    def llm_enabled(self) -> bool:
        return bool(self.api_key) and os.getenv("ADVISOR_DISABLE_LLM", "") != "1"


settings = Settings()
