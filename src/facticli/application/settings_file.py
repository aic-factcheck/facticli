from __future__ import annotations

import argparse
import os
import tomllib
from pathlib import Path
from typing import Any

# Config-file layering modelled on Codex ``config.toml`` profiles and pi's
# settings precedence: CLI flag > selected profile > [defaults] > built-in
# default. Keys mirror the long CLI flag names with dashes replaced by
# underscores, so the file is discoverable from ``facticli check --help``.

DEFAULT_CONFIG_FILENAMES: tuple[str, ...] = ("facticli.toml", ".facticli.toml")
CONFIG_ENV_VAR = "FACTICLI_CONFIG"
PROFILE_ENV_VAR = "FACTICLI_PROFILE"

# Table-valued keys accepted as a convenience and expanded to repeatable flags.
_TABLE_KEYS: dict[str, str] = {
    "stage_models": "stage_model",
    "stage_efforts": "stage_effort",
}
_LIST_KEYS: dict[str, str] = {
    "blocked_domains": "blocked_domain",
}


def find_config_path(explicit: str | None = None) -> Path | None:
    """Locate the config file: explicit path, env var, cwd, then user config dir."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Config file does not exist: {path}")
        return path
    env_path = os.getenv(CONFIG_ENV_VAR)
    if env_path:
        path = Path(env_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{CONFIG_ENV_VAR} points to a missing file: {path}")
        return path
    for name in DEFAULT_CONFIG_FILENAMES:
        candidate = Path.cwd() / name
        if candidate.is_file():
            return candidate
    xdg = os.getenv("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    candidate = Path(xdg) / "facticli" / "config.toml"
    if candidate.is_file():
        return candidate
    return None


def load_settings(path: Path, profile: str | None = None) -> dict[str, Any]:
    """Merge ``[defaults]`` with ``[profiles.<profile>]`` from a TOML file.

    Top-level keys outside those tables are treated as defaults too, so a
    minimal file can be flat. Unknown profiles raise ``ValueError``.
    """
    with path.open("rb") as handle:
        data = tomllib.load(handle)

    settings: dict[str, Any] = {}
    for key, value in data.items():
        if key in ("defaults", "profiles"):
            continue
        settings[key] = value
    defaults = data.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError(f"[defaults] in {path} must be a table.")
    settings.update(defaults)

    profiles = data.get("profiles", {})
    if profile:
        if not isinstance(profiles, dict) or profile not in profiles:
            available = ", ".join(sorted(profiles)) if isinstance(profiles, dict) and profiles else "none"
            raise ValueError(f"Profile {profile!r} not found in {path} (available: {available}).")
        selected = profiles[profile]
        if not isinstance(selected, dict):
            raise ValueError(f"[profiles.{profile}] in {path} must be a table.")
        settings.update(selected)
    return _expand_tables(settings)


def _expand_tables(settings: dict[str, Any]) -> dict[str, Any]:
    expanded: dict[str, Any] = {}
    for key, value in settings.items():
        normalized = key.replace("-", "_")
        if normalized in _TABLE_KEYS and isinstance(value, dict):
            expanded[_TABLE_KEYS[normalized]] = [f"{stage}={val}" for stage, val in value.items()]
        elif normalized in _LIST_KEYS and isinstance(value, list):
            expanded[_LIST_KEYS[normalized]] = [str(item) for item in value]
        else:
            expanded[normalized] = value
    return expanded


def _collect_defaults(parser: argparse.ArgumentParser, command: str | None) -> dict[str, Any]:
    """Defaults of the top-level parser plus those of the selected subcommand parser."""
    defaults: dict[str, Any] = {}
    for action in parser._actions:  # noqa: SLF001 - argparse exposes no public accessor
        if isinstance(action, argparse._SubParsersAction):
            if command is not None and command in action.choices:
                sub = action.choices[command]
                for sub_action in sub._actions:  # noqa: SLF001
                    if sub_action.dest not in ("help", argparse.SUPPRESS):
                        defaults[sub_action.dest] = sub_action.default
            continue
        if action.dest not in ("help", argparse.SUPPRESS):
            defaults[action.dest] = action.default
    return defaults


def apply_settings(
    args: argparse.Namespace,
    settings: dict[str, Any],
    parser: argparse.ArgumentParser,
) -> list[str]:
    """Fill namespace attributes still at their parser default from ``settings``.

    Returns the list of keys applied. Keys that do not correspond to an option
    of the active subcommand are ignored so one file can serve every command.
    An explicitly passed flag that equals the default is indistinguishable
    from an omitted one and is therefore overridden by the file.
    """
    defaults = _collect_defaults(parser, getattr(args, "command", None))
    applied: list[str] = []
    for key, value in settings.items():
        if key not in defaults or not hasattr(args, key):
            continue
        if getattr(args, key) != defaults[key]:
            continue
        setattr(args, key, value)
        applied.append(key)
    return applied
