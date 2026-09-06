"""Fetch the LoCoMo dataset.

Not vendored: it is ~2.8MB and carries its own licence terms, so it is fetched
on demand into data/ (gitignored) rather than committed here.

    python scripts/fetch_locomo.py

Source: https://github.com/snap-research/locomo
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DEST = Path("data/locomo10.json")


def main() -> int:
    if DEST.exists():
        print(f"{DEST} already present ({DEST.stat().st_size:,} bytes)")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {URL}")
    try:
        urllib.request.urlopen(URL, timeout=120)  # noqa: S310 - fixed https URL
    except Exception as exc:
        print(f"failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    urllib.request.urlretrieve(URL, DEST)  # noqa: S310 - fixed https URL
    print(f"wrote {DEST} ({DEST.stat().st_size:,} bytes)")
    print("Licence and citation: https://github.com/snap-research/locomo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
