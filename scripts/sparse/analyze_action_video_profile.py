#!/usr/bin/env python3
"""Replay a complete native profile on CPU and export typed proxy comparisons."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dreamwam.sparse.profile.analysis import analyze_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--read-count", type=int, default=56)
    args = parser.parse_args()
    result = analyze_profile(args.capture, args.out_dir, read_count=args.read_count)
    print(f"{result['status']}: {result['attention_records']} attention records, {result['token_rows']} token rows; no SR/speed claim")


if __name__ == "__main__":
    main()
