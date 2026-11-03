#!/usr/bin/env python3
"""
Rate-limited streaming replay DEMO -- not a benchmarking tool.

Replays a small slice of the streamed half of creditcard.csv at a
configurable transactions/sec rate, printing each transaction's row id and
true label as it "arrives," then scores the whole small batch with the
serial binary at the end so you can see the wall-clock cost of the
detector against a simulated arrival process.

This exists purely to demonstrate the "streaming" framing from the
proposal end-to-end. It is deliberately NOT used for any performance
number in docs/06_results/ -- see docs/01_dataset.md for why: the raw
Time column's real inter-arrival gaps span from sub-second to tens of
thousands of seconds, so pacing a full replay to real arrival times would
take ~2 days of wall clock. All actual benchmark numbers use --score-n on
the compiled binaries directly (throughput mode: process back-to-back, as
fast as possible).

Usage:
    python3 scripts/replay_demo.py --rate 20 --n 50
"""
import argparse
import csv
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--rate", type=float, default=10.0, help="transactions/sec to simulate")
    ap.add_argument("--n", type=int, default=30, help="number of streamed transactions to replay")
    args = ap.parse_args()

    with open(args.data) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    n_train = int(len(rows) * args.train_frac)
    stream_rows = rows[n_train:n_train + args.n]

    interval = 1.0 / args.rate
    print(f"Replaying {len(stream_rows)} transactions at {args.rate} tx/sec "
          f"(interval={interval*1000:.1f}ms)...")
    for i, row in enumerate(stream_rows):
        cls = row[-1].strip('"')
        amount = row[-2]
        print(f"  [t={i*interval:6.2f}s] row_id={n_train+i} amount={amount} "
              f"true_label={'FRAUD' if cls == '1' else 'normal'}")
        if i < len(stream_rows) - 1:
            time.sleep(interval)
    print("Replay complete. Score this batch for real with, e.g.:")
    # --score-n counts streamed rows (from the train/stream boundary), not absolute row ids.
    print(f"  ./build/serial --data {args.data} --score-n {args.n} "
          f"--adaptive --explain integrated --out /tmp/demo_out.csv")


if __name__ == "__main__":
    main()
