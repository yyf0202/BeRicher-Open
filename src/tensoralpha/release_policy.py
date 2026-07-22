"""Repository-level privacy and artifact policy used before publication."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_SKIP_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "build",
    "dist",
}
_FORBIDDEN_DIRECTORIES = {
    ".claude",
    ".codebuddy",
    ".cursor",
    "archive",
    "logs",
    "outputs",
    "paper_trading_data",
    "saved_models",
}
_FORBIDDEN_EXTENSIONS = {
    ".ckpt",
    ".key",
    ".pem",
    ".pt",
    ".pth",
    ".safetensors",
}
_CREDENTIAL_PATTERNS = (
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._~-]{20,}", re.IGNORECASE),
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:api[_-]?key|token|password|passwd|secret)\b\s*[:=]\s*[\"']([^\"']{12,})[\"']"
)
_EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_WINDOWS_ABSOLUTE = re.compile(r"(?i)(?<![A-Za-z0-9])[A-Z]:[\\/][^\s\"']+")
_HOME_ABSOLUTE = re.compile(r"(?<![\w])/(?:home|Users)/[^\s\"']+")
_PLACEHOLDER = re.compile(r"(?i)^(?:example|placeholder|replace|your|not-a-real|x{4,}|\$\{|<)")
_GITHUB_NOREPLY = re.compile(r"(?i)^[^@\s]+@users\.noreply\.github\.com$")


@dataclass(frozen=True, slots=True)
class PolicyFinding:
    kind: str
    path: str
    line: int | None = None


def _text_findings(path: Path, relative: str) -> list[PolicyFinding]:
    try:
        payload = path.read_bytes()
    except OSError:
        return [PolicyFinding("unreadable-file", relative)]
    if b"\0" in payload[:8192]:
        return []
    text = payload.decode("utf-8", errors="ignore")
    findings: list[PolicyFinding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if any(pattern.search(line) for pattern in _CREDENTIAL_PATTERNS):
            findings.append(PolicyFinding("credential", relative, line_number))
        assignment = _SECRET_ASSIGNMENT.search(line)
        if assignment and not _PLACEHOLDER.search(assignment.group(1)):
            findings.append(PolicyFinding("secret-assignment", relative, line_number))
        if _EMAIL.search(line):
            findings.append(PolicyFinding("email-address", relative, line_number))
        if _WINDOWS_ABSOLUTE.search(line) or _HOME_ABSOLUTE.search(line):
            findings.append(PolicyFinding("absolute-path", relative, line_number))
    return findings


def _git_history_findings(repository: Path) -> list[PolicyFinding]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "log", "HEAD", "--format=%H%x00%ae%x00%ce"],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return []
    if result.returncode != 0:
        return []

    findings: list[PolicyFinding] = []
    for line in result.stdout.splitlines():
        commit_hash, *emails = line.split("\0")
        if any(email and not _GITHUB_NOREPLY.fullmatch(email) for email in set(emails)):
            findings.append(PolicyFinding("git-email-address", f".git/commits/{commit_hash[:12]}"))
    return findings


def scan_repository(
    root: str | Path, *, max_file_bytes: int = 5 * 1024 * 1024
) -> list[PolicyFinding]:
    """Return metadata-only findings; never echo a matched secret value."""

    repository = Path(root).expanduser().resolve()
    findings: list[PolicyFinding] = []
    for path in sorted(repository.rglob("*")):
        relative_path = path.relative_to(repository)
        if any(part in _SKIP_DIRECTORIES for part in relative_path.parts):
            continue
        relative = relative_path.as_posix()
        if any(part in _FORBIDDEN_DIRECTORIES for part in relative_path.parts):
            findings.append(PolicyFinding("forbidden-path", relative))
        if not path.is_file():
            continue
        if path.suffix.lower() in _FORBIDDEN_EXTENSIONS:
            findings.append(PolicyFinding("forbidden-extension", relative))
        try:
            size = path.stat().st_size
        except OSError:
            findings.append(PolicyFinding("unreadable-file", relative))
            continue
        if size > max_file_bytes:
            findings.append(PolicyFinding("large-file", relative))
            continue
        findings.extend(_text_findings(path, relative))
    findings.extend(_git_history_findings(repository))
    return sorted(findings, key=lambda item: (item.path, item.line or 0, item.kind))
