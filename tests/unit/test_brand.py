import importlib
import subprocess
import sys
from pathlib import Path


def test_tensoralpha_package_and_cli_are_available():
    module = importlib.import_module("tensoralpha")

    result = subprocess.run(
        [sys.executable, "-m", "tensoralpha", "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert module.__version__ == "0.1.0"
    assert result.returncode == 0, result.stderr
    assert "usage: tensoralpha" in result.stdout


def test_tensoralpha_environment_prefix(monkeypatch):
    from tensoralpha.settings import Settings

    monkeypatch.setenv("TENSORALPHA_DATA_DIR", "runtime/tensor-data")

    settings = Settings.from_env(Path.cwd())

    assert settings.data_dir == (Path.cwd() / "runtime/tensor-data").resolve()


def test_project_metadata_uses_tensoralpha_identity():
    repository = Path(__file__).resolve().parents[2]
    metadata = (repository / "pyproject.toml").read_text(encoding="utf-8")

    assert 'name = "tensoralpha"' in metadata
    assert 'tensoralpha = "tensoralpha.cli:main"' in metadata
    assert "github.com/yyf0202/TensorAlpha" in metadata


def test_public_tree_has_no_legacy_brand():
    repository = Path(__file__).resolve().parents[2]
    legacy = "be" + "richer"
    skipped = {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "artifacts",
        "dist",
    }
    findings: list[str] = []

    for path in repository.rglob("*"):
        relative = path.relative_to(repository)
        if any(part in skipped for part in relative.parts):
            continue
        if legacy in relative.as_posix().lower():
            findings.append(relative.as_posix())
            continue
        if path.is_file() and path.stat().st_size <= 5 * 1024 * 1024:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if legacy in text.lower():
                findings.append(relative.as_posix())

    assert findings == []
