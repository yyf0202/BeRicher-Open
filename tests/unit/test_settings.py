from pathlib import Path

import pytest


def test_settings_default_to_project_relative_runtime_directories(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TENSORALPHA_TUSHARE_TOKEN", raising=False)

    from tensoralpha.settings import Settings

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.project_root == tmp_path.resolve()
    assert settings.data_dir == tmp_path.resolve() / "data"
    assert settings.model_dir == tmp_path.resolve() / "artifacts" / "models"
    assert settings.output_dir == tmp_path.resolve() / "artifacts" / "outputs"
    assert settings.paper_dir == tmp_path.resolve() / "artifacts" / "paper"
    assert settings.tushare_token is None


def test_network_data_source_requires_explicit_token(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TENSORALPHA_TUSHARE_TOKEN", raising=False)

    from tensoralpha.settings import ConfigurationError, Settings

    settings = Settings.from_env(project_root=tmp_path)

    with pytest.raises(ConfigurationError, match="TENSORALPHA_TUSHARE_TOKEN"):
        settings.require_tushare_token()


def test_environment_paths_are_resolved_without_changing_process_cwd(tmp_path: Path, monkeypatch):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    monkeypatch.setenv("TENSORALPHA_DATA_DIR", "runtime/market-data")

    from tensoralpha.settings import Settings

    original_cwd = Path.cwd()
    settings = Settings.from_env(project_root=project_root)

    assert Path.cwd() == original_cwd
    assert settings.data_dir == project_root.resolve() / "runtime" / "market-data"
