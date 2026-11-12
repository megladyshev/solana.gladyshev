"""Merge daily CSV files from one folder into another: same-day files are combined, duplicate rows dropped,
rows sorted by time. Standard library only, so it runs anywhere.

    python liq/merge.py SRC_DIR DST_DIR
"""
import csv
import gzip
import sys
from pathlib import Path


def _rows(f):
    with (gzip.open(f, "rt", encoding="utf-8", newline="") if f.suffix == ".gz" else f.open(encoding="utf-8", newline="")) as h:
        r = list(csv.reader(h))
    return (r[0], r[1:]) if r else (None, [])


def merge(src, dst):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    changed = 0
    for f in sorted(src.glob("*_20*.csv*")):
        out = dst / f.name.removesuffix(".gz")
        head, new = _rows(f)
        if head is None:
            continue
        old = _rows(out)[1] if out.exists() else []
        rows = sorted({tuple(x) for x in old + new if x}, key=lambda x: int(x[0]))
        if len(rows) != len(old):
            with out.open("w", encoding="utf-8", newline="") as h:
                w = csv.writer(h)
                w.writerow(head)
                w.writerows(rows)
            changed += 1
    return changed


if __name__ == "__main__":
    print(f"{merge(sys.argv[1], sys.argv[2])} file(s) updated")
