import argparse
import glob
import json
import os

import numpy as np

METRICS = ["dp", "abcc", "mcdp0", "eopp", "eodd", "ap", "f1", "acc"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="?", default="results")
    a = p.parse_args()
    runs = {}
    for f in sorted(glob.glob(os.path.join(a.results, "*", "*", "seed*.json"))):
        r = json.load(open(f))
        name = os.path.relpath(os.path.dirname(f), a.results)
        for part in ("test", "transfer"):
            if part in r:
                runs.setdefault(f"{name} ({part})" if "transfer" in r else name, []).append(r[part])
    print("run".ljust(36) + "".join(m.rjust(13) for m in METRICS) + "n".rjust(4))
    for name, rows in runs.items():
        cells = []
        for m in METRICS:
            v = [r[m] for r in rows if m in r]
            cells.append(f"{np.mean(v):.1f}±{np.std(v):.1f}" if v else "-")
        print(name.ljust(36) + "".join(c.rjust(13) for c in cells) + str(len(rows)).rjust(4))


if __name__ == "__main__":
    main()
