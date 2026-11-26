#!/usr/bin/env python3
"""
Collect every figure, table and key result file into a single zip package
ready for journal submission / supplementary material.
"""

import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"
RES_DIR = ROOT / "results"
OUT_ZIP = ROOT / "journal_package_extended.zip"

EXTRA_FILES = [
    "test_predictions.csv",
    "training_history.csv",
    "summary_test.csv",
    "test_metrics_ci.json",
    "journal_captions_extended.md",
]


def main():
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for folder, arcname in [(FIG_DIR, "figures"), (TAB_DIR, "tables")]:
            if not folder.exists():
                continue
            for fn in sorted(folder.iterdir()):
                if fn.is_file():
                    z.write(fn, arcname + "/" + fn.name)

        for fn in EXTRA_FILES:
            fp = RES_DIR / fn
            if fp.exists():
                z.write(fp, fn)

    size_mb = OUT_ZIP.stat().st_size / 1e6
    print(f"Created: {OUT_ZIP} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
