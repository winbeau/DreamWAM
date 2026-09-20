#!/usr/bin/env python3
"""Summarize hash-verified raw-profile analysis; optional standalone figures."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dreamwam.sparse.profile.summary import summarize_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    result = summarize_analysis(args.analysis, args.out_dir, plot=args.plot)
    print(f"{result['status']}: {result['input_count']} observations; no independent-trial inference")


if __name__ == "__main__":
    main()
