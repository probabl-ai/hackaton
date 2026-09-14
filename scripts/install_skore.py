#!/usr/bin/env python3
"""Install skore-cli, write ``.skore``, and install lab skills.

Does not write ``.bob/mcp.json`` or launch Bob.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = "bob-ide"
SKORE_CLI_SPEC = "skore-cli>=0.4"
_REEXEC_ENV = "_INSTALL_SKORE_REEXEC"


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run ``cmd`` from the repo root. Capture output so students do not see it."""
    completed = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        if detail:
            print(detail, file=sys.stderr)
        raise SystemExit(completed.returncode)
    return completed


def ensure_skore_cli() -> None:
    """Install or upgrade skore-cli with pipx."""
    pipx = shutil.which("pipx")
    if pipx is None:
        raise SystemExit("pipx is not on PATH.")
    if shutil.which("skore") is None:
        _run([pipx, "install", SKORE_CLI_SPEC])
        return
    upgraded = _run([pipx, "upgrade", "skore-cli"], check=False)
    if upgraded.returncode != 0:
        _run([pipx, "install", "--force", SKORE_CLI_SPEC])


def install_skills() -> None:
    """Install the lab skills catalog for Bob IDE."""
    skore = shutil.which("skore")
    if skore is None:
        raise SystemExit("skore is not on PATH after installing skore-cli.")
    _run(
        [
            skore,
            "skills",
            "install",
            "all",
            "--repo",
            "probabl-ai/skills-hackaton",
            "--agent",
            HARNESS,
        ]
    )


def _reexec_with_skore_cli() -> None:
    """Re-run this file with the interpreter that has ``skore_cli``."""
    skore = shutil.which("skore")
    if skore is None:
        raise SystemExit("skore is not on PATH after installing skore-cli.")
    python = Path(skore).resolve().parent / "python"
    if not python.is_file():
        raise SystemExit(f"could not find the skore-cli interpreter at {python}")
    venv_root = python.parent.parent
    if Path(sys.prefix).resolve() == venv_root.resolve():
        raise SystemExit("skore-cli is not importable in its own interpreter.")
    os.environ[_REEXEC_ENV] = "1"
    os.execv(str(python), [str(python), *sys.argv])


def api_hub_url(uri: str) -> str:
    """Map a frontend hub URL to the API base URL.

    ``https://x.skore.probabl.ai`` → ``https://x.api.skore.probabl.ai``.
    """
    parsed = urlparse(uri.strip())
    host = parsed.netloc
    if not host:
        raise SystemExit(f"hub URI is not a URL: {uri!r}")
    if host.startswith("api.") or ".api." in host:
        return uri.rstrip("/")
    if host == "skore.probabl.ai":
        host = "api.skore.probabl.ai"
    elif host.endswith(".skore.probabl.ai"):
        tenant = host[: -len(".skore.probabl.ai")]
        host = f"{tenant}.api.skore.probabl.ai"
    return urlunparse(parsed._replace(netloc=host)).rstrip("/")


def write_skore() -> None:
    """Save ``.skore`` via skore-cli, then exit before harness config."""
    import click
    from skore_cli._hub_auth import ensure_login
    from skore_cli._skore import auth as _auth
    from skore_cli._skore import resolve_hub_uri
    from skore_cli.agent import _client
    from skore_cli.agent._commands import _create_workspace_api_key
    from skore_cli.agent._skore_file import SkoreConfig

    dest = REPO_ROOT / ".skore"
    try:
        config = SkoreConfig.load(REPO_ROOT)
        if config is not None and config.api_key and config.workspace:
            return

        os.environ["SKORE_HUB_URI"] = api_hub_url("https://ibm.skore.probabl.ai")
        hub_url = resolve_hub_uri(os.environ["SKORE_HUB_URI"], _auth)
        token = ensure_login(timeout=600)
        user_id, memberships = _client.me(hub_url, token)
        if not memberships:
            raise SystemExit(
                "You are not in a Hub workspace. One teammate creates a "
                "workspace on https://ibm.skore.probabl.ai named like your "
                "Kaggle team, invites the others, then you re-run this script."
            )
        if len(memberships) != 1:
            raise SystemExit(
                "Expected exactly one Hub workspace. This lab uses one "
                "workspace per Kaggle team. Leave extra workspaces, then "
                "re-run. See docs/GUIDED.md § 3."
            )
        membership = memberships[0]
        api_key = _create_workspace_api_key(
            hub_url, token, user_id, membership, HARNESS
        )
        SkoreConfig(
            hub_url=hub_url,
            workspace=membership.public_id,
            workspace_id=membership.workspace_id,
            api_key=api_key,
            harness=HARNESS,
        ).save(REPO_ROOT)
        print(f"wrote {dest} for workspace {membership.public_id!r}")
    except click.Abort:
        raise SystemExit(1) from None
    except click.ClickException as exc:
        raise SystemExit(exc.format_message()) from exc


def main() -> None:
    """Install skore-cli, write ``.skore``, install skills."""
    inner = os.environ.pop(_REEXEC_ENV, None) == "1"
    if not inner:
        ensure_skore_cli()
        try:
            import skore_cli  # noqa: F401
        except ImportError:
            _reexec_with_skore_cli()
            return
    try:
        write_skore()
    except ImportError:
        raise SystemExit("skore-cli is not importable.") from None
    install_skills()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
