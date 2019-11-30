#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd


def main(args):
    ldata = pd.read_csv(args.input_labeled)
    udata = pd.read_csv(args.input_unlabeled, names=["Tweet"], sep="\t")

    if args.sample_size > 1:
        udata = udata.sample(
            n=int(args.sample_size),
            random_state=args.random_seed
        ).reset_index(drop=True)
    elif args.sample_size < 1:
        udata = udata.sample(
            frac=args.sample_size,
            random_state=args.random_seed
        ).reset_index(drop=True)

    udata.insert(0, "ID", udata.index)
    udata["ID"] += ldata.shape[0]
    udata["Stance"] = "UNK"
    udata["Split"] = "Unlabel"

    pd.concat([ldata, udata], ignore_index=True).to_csv(
        args.output,
        index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_labeled")
    parser.add_argument("input_unlabeled")
    parser.add_argument("output")
    parser.add_argument("--random-seed", default=42, type=int)
    parser.add_argument("--sample-size", default=5000, type=float)

    args = parser.parse_args()

    main(args)
