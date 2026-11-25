#!/usr/bin/env python3
"""Download the AVerImaTeC dev split from Hugging Face and convert it for the AVeriTeC runner.

Writes ``dev.json`` (all 152 dev claims, ``claim_id`` = dev row index) and
``pilot_20.json`` (every non-Refuted claim plus 9 evenly spaced Refuted ones,
the image pilot in TODO.md) with the fields the runner and the scorer read:
``claim``, ``claim_date``, ``label``, ``images``.

Image names follow the dataset's ``images.zip`` with ``#`` replaced by ``_``.
With ``--image-url-base`` each name becomes a URL under that base, so the
runner downloads the images and reverse image search can fetch them; without
it, pass the unzipped folder to the runner with ``--image-dir``.

The data is not redistributed in this repository; see the dataset page for
its terms: https://huggingface.co/datasets/Rui4416/AVerImaTeC
"""

from __future__ import annotations

import argparse
import ast
import json
import urllib.request
from pathlib import Path
from typing import Any

DEV_URL = "https://huggingface.co/datasets/Rui4416/AVerImaTeC/resolve/main/val.json"
PILOT_REFUTED = 9


def _as_list(value: Any) -> list[Any]:
    # The release stores lists and dicts as Python reprs inside strings.
    if isinstance(value, str):
        return ast.literal_eval(value) if value.strip() else []
    return value or []


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, str) and value.startswith("{"):
        return ast.literal_eval(value)
    return value if isinstance(value, dict) else {}


def convert(rows: list[dict[str, Any]], image_url_base: str | None) -> list[dict[str, Any]]:
    converted = []
    for index, row in enumerate(rows):
        names = [name.replace("#", "_") for name in _as_list(row["claim_images"])]
        if image_url_base:
            names = [f"{image_url_base.rstrip('/')}/{name}" for name in names]
        metadata = _metadata(row.get("metadata"))
        converted.append(
            {
                "claim_id": index,
                "claim": row["claim_text"],
                "claim_date": row["date"],
                "label": row["label"],
                "images": names,
                "justification": row["justification"],
                "fact_checking_article": row["article"],
                "speaker": metadata.get("speaker"),
                "reporting_source": metadata.get("reporting_source"),
            }
        )
    return converted


def pilot(dev: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """All non-Refuted claims plus ``PILOT_REFUTED`` evenly spaced Refuted ones, in dev order."""
    refuted = [row for row in dev if row["label"] == "Refuted"]
    step = len(refuted) / PILOT_REFUTED
    chosen = [row for row in dev if row["label"] != "Refuted"]
    chosen += [refuted[int(k * step)] for k in range(PILOT_REFUTED)]
    return sorted(chosen, key=lambda row: row["claim_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", default="data/averimatec", help="Where to write dev.json and pilot_20.json.")
    parser.add_argument(
        "--image-url-base",
        default=None,
        help="Public base URL of the images (file names with '#' replaced by '_'); stores image URLs instead of names.",
    )
    args = parser.parse_args()

    with urllib.request.urlopen(DEV_URL, timeout=60) as response:
        rows = json.loads(response.read().decode("utf-8"))
    dev = convert(rows, args.image_url_base)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, records in (("dev.json", dev), ("pilot_20.json", pilot(dev))):
        (output / name).write_text(json.dumps(records, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {len(records)} claims to {output / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
