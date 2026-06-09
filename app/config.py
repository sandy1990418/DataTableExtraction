from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (DataTableExtraction/), resolved relative to this
# file so it works regardless of the process's current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class Settings:
    def __init__(self, **overrides) -> None:
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        self.OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")
        self.OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", os.getenv("AI_MODEL", "gpt-4o-mini"))
        self.MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "-1"))
        self.TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))
        self.DOWNLOAD_TTL_SECONDS: int = int(os.getenv("DOWNLOAD_TTL_SECONDS", "600"))
        self.SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", "1800"))
        for key, value in overrides.items():
            setattr(self, key, value)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
