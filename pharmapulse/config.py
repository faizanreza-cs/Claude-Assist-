"""Runtime settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    db_path: Path = field(
        default_factory=lambda: Path(_env("PHARMAPULSE_DB", "data/pharmapulse.duckdb"))
    )
    # "auto" picks anthropic when credentials exist, else mock. Also: anthropic | openai | mock
    provider: str = field(default_factory=lambda: _env("PHARMAPULSE_PROVIDER", "auto"))
    # model id for the anthropic/openai providers; required for those, unused in mock mode
    model: str = field(default_factory=lambda: _env("PHARMAPULSE_MODEL", ""))
    effort: str = field(default_factory=lambda: _env("PHARMAPULSE_EFFORT", "medium"))
    max_agent_steps: int = field(default_factory=lambda: int(_env("PHARMAPULSE_MAX_STEPS", "8")))
    sql_row_limit: int = field(default_factory=lambda: int(_env("PHARMAPULSE_SQL_LIMIT", "200")))
    # opt in to Anthropic server-side refusal fallback (models that support it)
    fallbacks: bool = field(
        default_factory=lambda: _env("PHARMAPULSE_FALLBACKS", "0").lower() in {"1", "true", "yes"}
    )


def get_settings() -> Settings:
    return Settings()
