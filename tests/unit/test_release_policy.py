import subprocess
import sys
from pathlib import Path


def test_release_policy_detects_credentials_and_personal_paths(tmp_path: Path):
    from tensoralpha.release_policy import scan_repository

    credential = "gh" + "p_" + "A" * 36
    personal_path = "D:" + "\\private\\workspace\\model.pt"
    (tmp_path / "bad.txt").write_text(
        f"credential={credential}\npath={personal_path}\n", encoding="utf-8"
    )

    findings = scan_repository(tmp_path)
    kinds = {finding.kind for finding in findings}

    assert "credential" in kinds
    assert "absolute-path" in kinds


def test_release_policy_detects_forbidden_artifacts(tmp_path: Path):
    from tensoralpha.release_policy import scan_repository

    model_dir = tmp_path / "saved_models"
    model_dir.mkdir()
    (model_dir / "weights.pt").write_bytes(b"weights")

    findings = scan_repository(tmp_path)

    assert any(finding.kind == "forbidden-path" for finding in findings)
    assert any(finding.kind == "forbidden-extension" for finding in findings)


def test_release_policy_detects_personal_email_in_git_history(tmp_path: Path):
    from tensoralpha.release_policy import scan_repository

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Example"], check=True)
    personal_email = "private" + "@" + "example.invalid"
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", personal_email],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "--allow-empty", "-q", "-m", "initial"],
        check=True,
    )

    findings = scan_repository(tmp_path)

    assert any(finding.kind == "git-email-address" for finding in findings)


def test_public_repository_passes_its_own_release_policy():
    from tensoralpha.release_policy import scan_repository

    repository = Path(__file__).resolve().parents[2]

    assert scan_repository(repository) == []


def test_release_check_script_runs_before_package_installation():
    repository = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [sys.executable, "-S", str(repository / "scripts" / "check_release.py"), "--help"],
        cwd=repository,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
