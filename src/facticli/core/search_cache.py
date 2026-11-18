from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Retrieval is the noisiest part of a fact-check run: the same query returns
# different results an hour later, so two pipeline variants that should be
# compared on equal terms end up judging different evidence. Recording
# provider responses and replaying them on a later identical request holds
# retrieval constant wherever the queries coincide.
#
# What is stored is the result list before per-run filtering (domain
# blocklist) and truncation, so a replay under different limits still starts
# from the same retrieval. Anything that changes what the provider is asked
# (query, count, locale, date cutoff) must be part of the request.


class SearchCache:
    """Directory of recorded search responses, replayed on identical requests."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory).expanduser()
        # Fail at startup, not at the first search: a cache that cannot record
        # would leave the later "replay" run silently live.
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"Search cache directory {self.directory} cannot be created: {exc}") from exc
        if not os.access(self.directory, os.W_OK):
            raise ValueError(f"Search cache directory {self.directory} is not writable.")

    def _path(self, request: dict[str, Any]) -> Path:
        canonical = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return self.directory / key[:2] / f"{key}.json"

    def get(self, request: dict[str, Any]) -> Any | None:
        """The recorded response for this request, or None when there is none."""
        try:
            entry = json.loads(self._path(request).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None  # absent or unreadable: a miss, overwritten by the next put
        return entry.get("value") if isinstance(entry, dict) else None

    def put(self, request: dict[str, Any], value: Any) -> None:
        path = self._path(request)
        entry = {"request": request, "stored_at": datetime.now(timezone.utc).isoformat(), "value": value}
        try:
            path.parent.mkdir(exist_ok=True)
            # Write-then-rename so concurrent batch workers never read half a file.
            temp = path.with_suffix(f".{os.getpid()}.tmp")
            temp.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
            os.replace(temp, path)
        except OSError:
            pass  # disk filled mid-run: lose this record, keep the run
