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
    _add_layer(settings, {key: value for key, value in data.items() if key not in ("defaults", "profiles")})
    defaults = data.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError(f"[defaults] in {path} must be a table.")
    _add_layer(settings, defaults)

    profiles = data.get("profiles", {})
    if profile:
        if not isinstance(profiles, dict) or profile not in profiles:
            available = ", ".join(sorted(profiles)) if isinstance(profiles, dict) and profiles else "none"
            raise ValueError(f"Profile {profile!r} not found in {path} (available: {available}).")
        selected = profiles[profile]
        if not isinstance(selected, dict):
            raise ValueError(f"[profiles.{profile}] in {path} must be a table.")
        _add_layer(settings, selected)
    return settings


def _add_layer(settings: dict[str, Any], layer: dict[str, Any]) -> None:
    """Merge a later layer into ``settings``, keeping iteration order equal to precedence order.

    A key the layer sets moves to the end. ``apply_settings`` lets a later key
    override an earlier one, which is what makes a profile win over defaults
    even when the two spell the same option differently.
    """
    for key, value in layer.items():
        key = key.replace("-", "_")
        if key in _TABLE_KEYS and isinstance(value, dict):
            key, value = _TABLE_KEYS[key], [f"{stage}={val}" for stage, val in value.items()]
        settings.pop(key, None)
        settings[key] = value


def _config_keys(parser: argparse.ArgumentParser, command: str | None) -> dict[str, argparse.Action]:
    """Map every accepted config key to the option it sets, for the selected subcommand.

    A key is the long flag name with underscores (``search_results`` for
    ``--search-results``) or the attribute the flag is stored under
    (``search_results_per_query``, ``blocked_domains``).
    """
    actions: list[argparse.Action] = []
    for action in parser._actions:  # noqa: SLF001 - argparse exposes no public accessor
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            if command is not None and command in action.choices:
                actions.extend(action.choices[command]._actions)  # noqa: SLF001
        else:
            actions.append(action)
    actions = [action for action in actions if action.dest not in ("help", argparse.SUPPRESS)]
    keys = {action.dest: action for action in actions}
    for action in actions:
        for option in action.option_strings:
            if option.startswith("--"):
                keys.setdefault(option[2:].replace("-", "_"), action)
    return keys


def _validated(action: argparse.Action, key: str, value: Any) -> Any:
    """Check a file value the way the flag would be: on/off, the option's type, its choices."""
    if isinstance(action, argparse._AppendAction):  # noqa: SLF001
        # A lone value for a repeatable option is one item, not a list of characters.
        return [_validated_item(action, key, item) for item in (value if isinstance(value, list) else [value])]
    return _validated_item(action, key, value)


def _validated_item(action: argparse.Action, key: str, value: Any) -> Any:
    if isinstance(action, argparse._StoreTrueAction):  # noqa: SLF001
        if not isinstance(value, bool):
            raise ValueError(f"Config key {key!r}: expected true or false, got {value!r}.")
        return value
    if action.type is None:
        if not isinstance(value, str):
            raise ValueError(f"Config key {key!r}: expected a string, got {value!r}.")
    else:
        try:
            value = action.type(str(value))  # type: ignore[operator]
        except (argparse.ArgumentTypeError, ValueError) as exc:
            raise ValueError(f"Config key {key!r}: {exc}") from exc
    if action.choices is not None and value not in action.choices:
        raise ValueError(f"Config key {key!r}: {value!r} is not one of {', '.join(map(str, action.choices))}.")
    return value


def apply_settings(
    args: argparse.Namespace,
    settings: dict[str, Any],
    parser: argparse.ArgumentParser,
) -> list[str]:
    """Fill namespace attributes still at their parser default from ``settings``.

    Returns the list of keys applied. Keys that do not correspond to an option
    of the active subcommand are ignored so one file can serve every command;
    a value the option itself would reject raises ``ValueError``.
    An explicitly passed flag that equals the default is indistinguishable
    from an omitted one and is therefore overridden by the file.
    """
    keys = _config_keys(parser, getattr(args, "command", None))
    applied: list[str] = []
    from_file: set[str] = set()
    for key, value in settings.items():
        action = keys.get(key)
        if action is None or not hasattr(args, action.dest):
            continue
        # An explicit CLI flag wins. A value this function set itself does not:
        # settings arrive ordered defaults-then-profile, and the same option may
        # be spelled differently in the two layers, so the later key overrides.
        if getattr(args, action.dest) != action.default and action.dest not in from_file:
            continue
        setattr(args, action.dest, _validated(action, key, value))
        from_file.add(action.dest)
        applied.append(key)
    return applied
