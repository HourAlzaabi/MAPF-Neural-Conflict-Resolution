"""
make_demo_sets.py  -  build curated scenario folders for the live demo.

Scans a dataset (default: the held-out test set) and sorts scenarios into
buckets by how the neural policy does on them, so the demo always has reliable
examples of each case to show:

    demo_sets/1_cbs/            a handful of maps (CBS always solves these)
    demo_sets/2_cnn_crash/      CNN alone COLLIDES   (shows why we need a shield)
    demo_sets/3_cnn_ok/         CNN alone succeeds
    demo_sets/4_shield_saves/   CNN alone crashes, but CNN+Shield succeeds  <-- the money shot
    demo_sets/5_shield_ok/      CNN+Shield succeeds
    demo_sets/6_shield_deadlock/ CNN+Shield deadlocks (honest limitation)

Run it once before presenting (fast):   python make_demo_sets.py
Then the demo menu points the visualizer at these folders.
"""

import argparse
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "task4"))
import evaluate  # noqa: E402


def classify(res):
    if res["collisions"] > 0:
        return "COLLISION"
    if res["reached"]:
        return "SUCCESS"
    return "DEADLOCK"


def main():
    ap = argparse.ArgumentParser(description="Curate demo scenario sets.")
    ap.add_argument("--dataset", default=os.path.join(HERE, "task1", "wh_holdout"),
                    help="folder of scenario .json files (default: held-out set)")
    ap.add_argument("--model", default=os.path.join(HERE, "model.pt"))
    ap.add_argument("--out", default=os.path.join(HERE, "demo_sets"))
    ap.add_argument("--per", type=int, default=6, help="max scenarios per bucket")
    args = ap.parse_args()

    if not os.path.isdir(args.dataset):
        raise SystemExit("No dataset at %s - run make_holdout.py first." % args.dataset)
    files = sorted(glob.glob(os.path.join(args.dataset, "*.json")))
    if not files:
        raise SystemExit("No .json scenarios in %s" % args.dataset)

    print("Loading model and scanning %d scenarios ..." % len(files))
    pol = evaluate.make_policy("model", args.model)

    buckets = {k: [] for k in
               ("1_cbs", "2_cnn_crash", "3_cnn_ok",
                "4_shield_saves", "5_shield_ok", "6_shield_deadlock")}

    def full(name):
        return len(buckets[name]) >= args.per

    for f in files:
        s = evaluate.load_scenario(f)
        if not full("1_cbs"):
            buckets["1_cbs"].append(f)
        r0 = evaluate.rollout(s, pol, shield=None)     # CNN alone
        r1 = evaluate.rollout(s, pol, shield={})       # CNN + shield
        c0, c1 = classify(r0), classify(r1)
        if c0 == "COLLISION" and not full("2_cnn_crash"):
            buckets["2_cnn_crash"].append(f)
        if c0 == "SUCCESS" and not full("3_cnn_ok"):
            buckets["3_cnn_ok"].append(f)
        if c0 == "COLLISION" and c1 == "SUCCESS" and not full("4_shield_saves"):
            buckets["4_shield_saves"].append(f)
        if c1 == "SUCCESS" and not full("5_shield_ok"):
            buckets["5_shield_ok"].append(f)
        if c1 == "DEADLOCK" and not full("6_shield_deadlock"):
            buckets["6_shield_deadlock"].append(f)
        if all(full(k) for k in buckets):
            break

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    print("\n  bucket                 count  ->  %s/" % os.path.basename(args.out))
    for name, fs in buckets.items():
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        for f in fs:
            shutil.copy(f, d)
        print("  %-22s %3d" % (name, len(fs)))
    print("\nDone. The demo menu reads these folders.")


if __name__ == "__main__":
    main()
