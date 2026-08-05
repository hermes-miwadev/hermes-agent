"""Repo-local Git author identity inheritance for Hermes-managed coding
workspaces (fresh clones, worktrees).

Problem: a brand-new git clone or worktree created for a coding task has no
Git author identity of its own. Nothing before this module resolved one, so
the first commit blocked waiting for someone to configure ``git config``
interactively -- disruptive mid-task, and easy to miss until the commit
step actually runs.

This module only ever writes **repo-local** config (``git config --local``)
to the **newly created workspace**, and only when that workspace doesn't
already have a complete identity of its own -- a linked worktree normally
already has one, inherited automatically from the shared repository config
(``git config --local`` for a linked worktree reads/writes the *shared*
config in the common ``.git`` dir, not a per-worktree file, unless
``extensions.worktreeConfig`` is explicitly enabled -- so
:func:`has_local_identity` naturally returns True for those without any
extra code). Global (``--global``) and system config are never touched.

Precedence when the workspace doesn't already have an identity:

1. The source/parent repository's own **repo-local** ``user.name`` /
   ``user.email`` (the repo a clone or worktree was created *from*) --
   read with ``--local`` specifically, not git's normal fallthrough to
   global/system, so this tier only fires on an identity someone
   deliberately scoped to that repo.
2. The current process's **effective** Git identity -- a plain
   ``git config --get user.name`` / ``user.email`` (git's normal
   local -> global -> system resolution), i.e. whatever identity Hermes
   itself would already use if it tried to commit right now.
3. An explicit ``coding_workspace_identity: {name, email}`` in
   ``config.yaml`` -- an operator-configured last resort. Both fields must
   be set; nothing is invented or derived (e.g. never synthesised from a
   GitHub username).

If none of the three yields a complete identity, no fake value is ever
written -- :func:`ensure_workspace_identity` returns a result with
``applied=False`` and an operator-facing ``message`` explaining exactly how
to configure one, on the theory that surfacing this immediately at
workspace-creation time beats discovering it silently at the first blocked
commit. The message (and every other string this module produces) never
includes the resolved email address -- see the module docstring note in
:class:`GitIdentity`.
"""

from __future__ import annotations

import dataclasses
import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class GitIdentityError(Exception):
    """Raised internally when resolving or applying an identity fails.

    Never publicly propagated by :func:`ensure_workspace_identity` (which
    catches this and returns an :class:`IdentityResult` instead) -- callers
    that want an exception-based API can use :func:`resolve_identity`
    directly.
    """


@dataclasses.dataclass(frozen=True)
class GitIdentity:
    """A resolved (name, email) pair. Never constructed with invented values."""

    name: str
    email: str

    def __repr__(self) -> str:
        # A bare repr() (e.g. an uncaught exception's traceback, or a
        # debugger) must not leak the email into logs.
        return f"GitIdentity(name={self.name!r}, email='<redacted>')"


@dataclasses.dataclass(frozen=True)
class IdentityResult:
    """Outcome of :func:`ensure_workspace_identity`.

    ``message`` is set only when ``applied`` is False, and is always a
    static, operator-facing instructional string -- it never contains the
    resolved (or attempted) email address.
    """

    applied: bool
    source: str  # "already-configured" | "source-repo" | "process-effective" | "hermes-config" | "unavailable"
    message: Optional[str] = None


_NO_IDENTITY_MESSAGE = (
    "No trustworthy Git author identity is available for this workspace, so "
    "commits here will be blocked until one is configured. Fix this with "
    "either:\n"
    '  git config --local user.name "Your Name"\n'
    '  git config --local user.email "you@example.com"\n'
    "(run inside the new workspace), or set a fallback identity Hermes can "
    "reuse for future workspaces:\n"
    '  hermes config set coding_workspace_identity.name "Your Name"\n'
    '  hermes config set coding_workspace_identity.email "you@example.com"'
)


def _run_git(
    args: list[str], *, cwd: Optional[Path] = None, timeout: float = 10
) -> Optional[subprocess.CompletedProcess]:
    """Run a git subprocess. Returns None (never raises) on a subprocess-
    level failure (git missing, timeout, OS error) -- callers treat that
    the same as "no answer" from git, not a hard error."""
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("git identity: `git %s` failed: %s", " ".join(args), exc)
        return None


def _config_get(args: list[str], *, cwd: Optional[Path]) -> Optional[str]:
    result = _run_git(["config", *args], cwd=cwd)
    if result is None or result.returncode != 0:
        return None
    value = (result.stdout or "").strip()
    return value or None


def _read_local_identity(repo_path: Path) -> Optional[GitIdentity]:
    """Tier 1: *repo_path*'s own --local user.name/user.email, with no
    fallthrough to global/system config."""
    name = _config_get(["--local", "--get", "user.name"], cwd=repo_path)
    email = _config_get(["--local", "--get", "user.email"], cwd=repo_path)
    if name and email:
        return GitIdentity(name=name, email=email)
    return None


def _read_effective_identity(cwd: Optional[Path]) -> Optional[GitIdentity]:
    """Tier 2: the process's effective identity (git's normal local ->
    global -> system resolution) at *cwd*.

    Callers always pass an already-existing workspace directory here (by
    the time :func:`ensure_workspace_identity` runs, ``git worktree add``
    or ``git init`` has already created it) -- deliberately not falling
    back to the process's own ambient cwd if *cwd* doesn't exist, since
    that could pick up an unrelated repository's local identity that has
    nothing to do with the workspace being resolved for.
    """
    name = _config_get(["--get", "user.name"], cwd=cwd)
    email = _config_get(["--get", "user.email"], cwd=cwd)
    if name and email:
        return GitIdentity(name=name, email=email)
    return None


def _read_configured_hermes_identity() -> Optional[GitIdentity]:
    """Tier 3: the operator-configured ``coding_workspace_identity`` in
    config.yaml. Both fields must be explicitly set -- nothing here is
    invented or derived."""
    try:
        from hermes_cli.config import load_config

        config = load_config()
    except Exception:
        logger.debug("git identity: failed to load config for tier 3", exc_info=True)
        return None
    raw = (config or {}).get("coding_workspace_identity") or {}
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    email = str(raw.get("email") or "").strip()
    if name and email:
        return GitIdentity(name=name, email=email)
    return None


def has_local_identity(repo_path: Path) -> bool:
    """True when *repo_path* already has a complete --local identity.

    For a linked worktree this is naturally True whenever the shared
    parent repository has one configured -- ``--local`` for a linked
    worktree resolves to the repository's shared config, not a
    per-worktree file.
    """
    return _read_local_identity(repo_path) is not None


def resolve_identity(
    *, source_repo: Optional[Path] = None, cwd: Optional[Path] = None
) -> GitIdentity:
    """Resolve a trustworthy identity via the documented precedence.

    Raises:
        GitIdentityError: with an operator-facing (email-free) message if
            no tier yields a complete identity. Never invents one.
    """
    if source_repo is not None:
        identity = _read_local_identity(source_repo)
        if identity is not None:
            return identity

    identity = _read_effective_identity(cwd)
    if identity is not None:
        return identity

    identity = _read_configured_hermes_identity()
    if identity is not None:
        return identity

    raise GitIdentityError(_NO_IDENTITY_MESSAGE)


def _resolve_identity_with_source(
    *, source_repo: Optional[Path], cwd: Optional[Path]
) -> tuple[Optional[GitIdentity], str]:
    """Like resolve_identity, but also reports which tier answered (for
    IdentityResult.source) instead of raising."""
    if source_repo is not None:
        identity = _read_local_identity(source_repo)
        if identity is not None:
            return identity, "source-repo"

    identity = _read_effective_identity(cwd)
    if identity is not None:
        return identity, "process-effective"

    identity = _read_configured_hermes_identity()
    if identity is not None:
        return identity, "hermes-config"

    return None, "unavailable"


def _apply_local_identity(workspace: Path, identity: GitIdentity) -> None:
    """Write *identity* to *workspace* via ``git config --local`` only.

    Raises:
        GitIdentityError: on any subprocess failure (git missing, non-zero
            exit, timeout) -- surfaced safely, never silently swallowed.
    """
    for key, value in (("user.name", identity.name), ("user.email", identity.email)):
        result = _run_git(["config", "--local", key, value], cwd=workspace)
        if result is None:
            raise GitIdentityError(
                f"Failed to configure repo-local Git identity for {workspace} "
                f"(could not run git)."
            )
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            raise GitIdentityError(
                f"Failed to configure repo-local Git identity for {workspace}: {stderr}"
            )


def ensure_workspace_identity(
    workspace: Path, *, source_repo: Optional[Path] = None
) -> IdentityResult:
    """Ensure *workspace* has a repo-local Git author identity.

    No-ops (and never overwrites anything) if *workspace* already has a
    complete --local identity of its own -- including one inherited
    automatically as a linked worktree. Otherwise resolves one via
    :func:`resolve_identity` and applies it with ``git config --local``,
    scoped to *workspace* only; global/system config is never touched.

    Returns an :class:`IdentityResult` rather than raising, so callers
    (interactive CLI, autonomous kanban dispatch) can each decide how to
    surface a missing identity -- print, log, or otherwise -- without this
    module dictating an experience it can't know is right everywhere it's
    called from.
    """
    if has_local_identity(workspace):
        return IdentityResult(applied=True, source="already-configured")

    identity, source = _resolve_identity_with_source(source_repo=source_repo, cwd=workspace)
    if identity is None:
        return IdentityResult(applied=False, source="unavailable", message=_NO_IDENTITY_MESSAGE)

    try:
        _apply_local_identity(workspace, identity)
    except GitIdentityError as exc:
        return IdentityResult(applied=False, source="unavailable", message=str(exc))

    logger.info("Configured a repo-local Git author identity for %s (source: %s)", workspace, source)
    return IdentityResult(applied=True, source=source)
