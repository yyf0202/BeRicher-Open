"""Runtime configuration with safe, portable defaults.

Secrets are read only from environment variables.  Importing this module never
changes the process working directory and never creates files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(RuntimeError):
    """Raised when a requested capability is missing required configuration."""


def _resolve_path(project_root: Path, value: str | None, default: str) -> Path:
    path = Path(value or default).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    """Paths and optional credentials used by TensorAlpha."""

    project_root: Path
    data_dir: Path
    model_dir: Path
    output_dir: Path
    paper_dir: Path
    tushare_token: str | None = None

    @classmethod
    def from_env(cls, project_root: str | Path | None = None) -> Settings:
        root = Path(project_root or Path.cwd()).expanduser().resolve()
        token = os.getenv("TENSORALPHA_TUSHARE_TOKEN")
        token = token.strip() if token and token.strip() else None
        return cls(
            project_root=root,
            data_dir=_resolve_path(root, os.getenv("TENSORALPHA_DATA_DIR"), "data"),
            model_dir=_resolve_path(root, os.getenv("TENSORALPHA_MODEL_DIR"), "artifacts/models"),
            output_dir=_resolve_path(
                root, os.getenv("TENSORALPHA_OUTPUT_DIR"), "artifacts/outputs"
            ),
            paper_dir=_resolve_path(root, os.getenv("TENSORALPHA_PAPER_DIR"), "artifacts/paper"),
            tushare_token=token,
        )

    def require_tushare_token(self) -> str:
        """Return the configured token or explain how to configure it."""

        if self.tushare_token is None:
            raise ConfigurationError(
                "Tushare access requires TENSORALPHA_TUSHARE_TOKEN. "
                "Set it in your shell or an untracked .env file."
            )
        return self.tushare_token
