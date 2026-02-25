#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))


def _maybe_reexec_local_venv() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    venv_dir = repo_root / ".venv"
    venv_python = repo_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return
    if os.getenv("_FACTICLI_LOCAL_VENV_REEXEC") == "1":
        return
    active_venv = os.getenv("VIRTUAL_ENV")
    if active_venv and Path(active_venv).resolve() == venv_dir.resolve():
        return
    env = dict(os.environ)
    env["_FACTICLI_LOCAL_VENV_REEXEC"] = "1"
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["PATH"] = str(venv_dir / "bin") + os.pathsep + env.get("PATH", "")
    os.execve(str(venv_python), [str(venv_python), __file__, *sys.argv[1:]], env)


def main() -> int:
    _maybe_reexec_local_venv()
    _bootstrap_src_path()
    from facticli.averitec_submission import main as module_main

    return module_main()


if __name__ == "__main__":
    raise SystemExit(main())
