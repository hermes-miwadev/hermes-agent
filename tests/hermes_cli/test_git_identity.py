"""Tests for hermes_cli.git_identity.

Repo-local Git author identity inheritance for Hermes-managed coding
workspaces (fresh clones, worktrees). Every test hermetically isolates
--global/--system config via GIT_CONFIG_GLOBAL/GIT_CONFIG_SYSTEM (pointed
at an empty file) so the real machine's git config can never leak in or
be mistaken for a resolved identity -- and so these tests can assert
"nothing was inherited" deterministically regardless of what's actually
configured on the host running them.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli.git_identity import (
    GitIdentity,
    GitIdentityError,
    ensure_workspace_identity,
    has_local_identity,
    resolve_identity,
)


@pytest.fixture(autouse=True)
def _hermetic_git_config(tmp_path, monkeypatch):
    """No real --global/--system config can be seen by any git call here."""
    empty = tmp_path / "empty-gitconfig"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(empty))
    # Isolate the "operator-configured fallback" tier's config.yaml too.
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=check,
    )


def _init_repo(path: Path, *, identity: tuple[str, str] | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    if identity:
        name, email = identity
        _git(path, "config", "--local", "user.name", name)
        _git(path, "config", "--local", "user.email", email)
        (path / "README.md").write_text("x\n", encoding="utf-8")
        _git(path, "add", "README.md")
        _git(path, "-c", f"user.name={name}", "-c", f"user.email={email}", "commit", "-qm", "init")
    return path


def _fresh_workspace(path: Path) -> Path:
    """A brand-new, standalone repo with no identity of its own (models a
    freshly-cloned or newly-initialised workspace)."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    return path


def _set_config_yaml_fallback(tmp_path: Path, name: str, email: str) -> None:
    hermes_home = Path(__import__("os").environ["HERMES_HOME"])
    (hermes_home / "config.yaml").write_text(
        f'coding_workspace_identity:\n  name: "{name}"\n  email: "{email}"\n',
        encoding="utf-8",
    )


# ── has_local_identity / resolve_identity (pure logic) ──────────────────────


class TestHasLocalIdentity:
    def test_false_for_bare_repo(self, tmp_path):
        repo = _fresh_workspace(tmp_path / "bare")
        assert has_local_identity(repo) is False

    def test_true_when_both_fields_set(self, tmp_path):
        repo = _init_repo(tmp_path / "with-identity", identity=("Ada", "ada@example.com"))
        assert has_local_identity(repo) is True

    def test_false_when_only_name_set(self, tmp_path):
        repo = _fresh_workspace(tmp_path / "half")
        _git(repo, "config", "--local", "user.name", "Ada")
        assert has_local_identity(repo) is False

    def test_false_when_only_email_set(self, tmp_path):
        repo = _fresh_workspace(tmp_path / "half2")
        _git(repo, "config", "--local", "user.email", "ada@example.com")
        assert has_local_identity(repo) is False

    def test_per_command_dash_c_override_is_not_a_local_identity(self, tmp_path):
        """A `-c user.name=...` override used for a single commit must not
        be mistaken for a persisted --local identity (this is exactly how
        several existing test fixtures in this repo create commits without
        actually configuring an identity)."""
        repo = _fresh_workspace(tmp_path / "dash-c-only")
        (repo / "f.txt").write_text("x", encoding="utf-8")
        _git(repo, "add", "f.txt")
        _git(repo, "-c", "user.name=Ephemeral", "-c", "user.email=eph@example.com", "commit", "-qm", "x")
        assert has_local_identity(repo) is False


class TestResolveIdentityPrecedence:
    def test_tier1_source_repo_local_identity_wins(self, tmp_path):
        source = _init_repo(tmp_path / "source", identity=("Source Owner", "source@example.com"))
        identity = resolve_identity(source_repo=source, cwd=tmp_path / "dest")
        assert identity == GitIdentity(name="Source Owner", email="source@example.com")

    def test_tier2_process_effective_identity_when_no_source_repo_identity(self, tmp_path, monkeypatch):
        source = _fresh_workspace(tmp_path / "source-empty")
        # Simulate the process's own effective identity via a real --global
        # config pointed at by GIT_CONFIG_GLOBAL (still hermetic -- this is
        # a test-owned file, not the real machine's ~/.gitconfig).
        global_cfg = tmp_path / "process-global-gitconfig"
        global_cfg.write_text('[user]\n\tname = Process User\n\temail = process@example.com\n', encoding="utf-8")
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_cfg))

        dest = _fresh_workspace(tmp_path / "dest")
        identity = resolve_identity(source_repo=source, cwd=dest)
        assert identity == GitIdentity(name="Process User", email="process@example.com")

    def test_tier3_configured_hermes_identity_as_last_resort(self, tmp_path):
        source = _fresh_workspace(tmp_path / "source-empty2")
        _set_config_yaml_fallback(tmp_path, "Hermes Fallback", "hermes-fallback@example.com")

        identity = resolve_identity(source_repo=source, cwd=tmp_path / "dest")
        assert identity == GitIdentity(name="Hermes Fallback", email="hermes-fallback@example.com")

    def test_no_tier_available_raises_without_inventing_a_value(self, tmp_path):
        source = _fresh_workspace(tmp_path / "source-empty3")
        with pytest.raises(GitIdentityError):
            resolve_identity(source_repo=source, cwd=tmp_path / "dest")

    def test_source_repo_tier_reads_local_only_not_its_own_global_fallthrough(self, tmp_path, monkeypatch):
        """Tier 1 must use --local specifically: a source repo with no
        --local identity, even if IT would resolve one through its own
        global fallback, must not be treated as tier-1-satisfied (that
        value surfaces via tier 2 instead, applied to the *new* workspace's
        own context -- the distinction matters for auditability)."""
        source = _fresh_workspace(tmp_path / "source-global-only")
        global_cfg = tmp_path / "some-global-gitconfig"
        global_cfg.write_text('[user]\n\tname = Global Only\n\temail = global@example.com\n', encoding="utf-8")
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_cfg))

        assert has_local_identity(source) is False
        dest = _fresh_workspace(tmp_path / "dest")
        identity = resolve_identity(source_repo=source, cwd=dest)
        # Resolved (via tier 2, not tier 1) -- but let's confirm tier 1
        # alone (no cwd fallback) really finds nothing:
        from hermes_cli.git_identity import _read_local_identity
        assert _read_local_identity(source) is None
        assert identity == GitIdentity(name="Global Only", email="global@example.com")


# ── ensure_workspace_identity (the integration-facing entrypoint) ──────────


class TestEnsureWorkspaceIdentity:
    def test_fresh_clone_inherits_source_repo_local_identity(self, tmp_path):
        source = _init_repo(tmp_path / "source", identity=("Source Owner", "source@example.com"))
        dest = _fresh_workspace(tmp_path / "dest")

        result = ensure_workspace_identity(dest, source_repo=source)

        assert result.applied is True
        assert result.source == "source-repo"
        applied = subprocess.run(
            ["git", "-C", str(dest), "config", "--local", "--get", "user.name"],
            capture_output=True, text=True,
        )
        assert applied.stdout.strip() == "Source Owner"

    def test_destination_identity_is_preserved_not_overwritten(self, tmp_path):
        source = _init_repo(tmp_path / "source", identity=("Source Owner", "source@example.com"))
        dest = _init_repo(tmp_path / "dest", identity=("Already Set", "already@example.com"))

        result = ensure_workspace_identity(dest, source_repo=source)

        assert result.applied is True
        assert result.source == "already-configured"
        still_there = subprocess.run(
            ["git", "-C", str(dest), "config", "--local", "--get", "user.name"],
            capture_output=True, text=True,
        )
        assert still_there.stdout.strip() == "Already Set"

    def test_worktree_with_shared_identity_needs_no_mutation(self, tmp_path):
        """A linked worktree's --local config IS the parent repo's shared
        config -- ensure_workspace_identity must recognise this as already
        satisfied and not attempt any write."""
        source = _init_repo(tmp_path / "source", identity=("Shared Owner", "shared@example.com"))
        worktree = tmp_path / "wt"
        _git(source, "worktree", "add", "-b", "feature", str(worktree), "HEAD")

        assert has_local_identity(worktree) is True  # inherited automatically

        result = ensure_workspace_identity(worktree, source_repo=source)

        assert result.applied is True
        assert result.source == "already-configured"

    def test_incomplete_identity_is_rejected_not_applied(self, tmp_path):
        source = _fresh_workspace(tmp_path / "source-empty")
        dest = _fresh_workspace(tmp_path / "dest")

        result = ensure_workspace_identity(dest, source_repo=source)

        assert result.applied is False
        assert result.source == "unavailable"
        assert result.message is not None
        assert has_local_identity(dest) is False

    def test_no_global_git_config_is_ever_changed(self, tmp_path, monkeypatch):
        writable_global = tmp_path / "writable-global-gitconfig"
        writable_global.write_text("", encoding="utf-8")
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(writable_global))

        source = _init_repo(tmp_path / "source", identity=("Source Owner", "source@example.com"))
        dest = _fresh_workspace(tmp_path / "dest")

        ensure_workspace_identity(dest, source_repo=source)

        # The --global file must still be empty -- only dest's --local
        # config may have been written to.
        assert writable_global.read_text(encoding="utf-8") == ""

    def test_subprocess_failure_applying_identity_is_surfaced_safely(self, tmp_path, monkeypatch):
        """A git-config write failure (e.g. git being unrunnable, or a
        locked/unwritable config file) must be reported through
        IdentityResult, never raise an uncaught exception out of
        ensure_workspace_identity."""
        source = _init_repo(tmp_path / "source", identity=("Source Owner", "source@example.com"))
        dest = _fresh_workspace(tmp_path / "dest")

        import hermes_cli.git_identity as gi

        original_run_git = gi._run_git

        def _fake_run_git(args, *, cwd=None, timeout=10):
            # Let reads (--get, or anything on source_repo) through as
            # normal; only the WRITE to dest ("config --local user.X value")
            # fails, simulating e.g. git being unrunnable for that call.
            is_write = args[:2] == ["config", "--local"] and len(args) > 2 and args[2] not in ("--get",)
            if is_write and cwd == dest:
                return None
            return original_run_git(args, cwd=cwd, timeout=timeout)

        monkeypatch.setattr(gi, "_run_git", _fake_run_git)

        result = ensure_workspace_identity(dest, source_repo=source)

        assert result.applied is False
        assert result.message is not None
        assert "Failed to configure" in result.message
        # The write never happened -- no partial/corrupt identity left behind.
        assert has_local_identity(dest) is False

    def test_no_identity_values_leak_into_result_message(self, tmp_path):
        source = _fresh_workspace(tmp_path / "source-empty")
        dest = _fresh_workspace(tmp_path / "dest")

        result = ensure_workspace_identity(dest, source_repo=source)

        assert result.message is not None
        # The instructional message only ever uses the generic placeholder
        # "you@example.com" -- never a real resolved/attempted value (there
        # is none in this scenario since no tier resolved an identity). Every
        # "@example.com" occurrence must be part of that literal placeholder.
        assert "you@example.com" in result.message
        assert result.message.count("@example.com") == result.message.count("you@example.com")

    def test_no_identity_values_leak_into_repr(self, tmp_path):
        source = _init_repo(tmp_path / "source", identity=("Source Owner", "source@example.com"))
        identity = resolve_identity(source_repo=source, cwd=tmp_path / "dest")
        assert "source@example.com" not in repr(identity)
        assert "<redacted>" in repr(identity)
