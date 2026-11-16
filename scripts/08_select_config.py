#!/usr/bin/env python
"""Step 8: configuration selection rule (Appendix B.3).

Input: a CSV with one row per candidate configuration and the columns
    config, vul_ratio, pass1
measured on the development data (ProSec prompts held out from the 40% adaptation
split for `vul_ratio`; the utility benchmark for `pass1`), plus the base model's
pass@1 on the same utility data.

Rule: keep configurations with pass1 >= min_relative_pass1 * base_pass1, choose the
lowest vul_ratio, break ties by the highest pass1."""

import argparse
import csv


def select(rows, base_pass1, min_relative_pass1=0.90):
    ok = [r for r in rows if float(r["pass1"]) >= min_relative_pass1 * base_pass1]
    if not ok:
        return None
    return min(ok, key=lambda r: (float(r["vul_ratio"]), -float(r["pass1"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--base_pass1", type=float, required=True)
    ap.add_argument("--min_relative_pass1", type=float, default=0.90)
    args = ap.parse_args()
    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    best = select(rows, args.base_pass1, args.min_relative_pass1)
    if best is None:
        print("no configuration satisfies the utility constraint")
    else:
        print(f"selected: {best['config']}  vul_ratio={best['vul_ratio']}  pass1={best['pass1']}")


if __name__ == "__main__":
    main()
