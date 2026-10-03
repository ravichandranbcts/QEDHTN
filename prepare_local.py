#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_local.py  (NO Kaggle token needed)
===========================================
After you download the datasets in your browser and unzip them anywhere inside
this folder, run:

    python prepare_local.py

It searches this folder (recursively) for the raw files and writes the three
CSVs the pipeline expects, into ./data/ :

    data/creditcard.csv   (UCI/ULB, label 'Class')   <- if found
    data/paysim.csv       (PaySim,  label 'isFraud')  <- if found
    data/ieee_cis.csv     (IEEE-CIS, label 'isFraud') <- merges transaction+identity

It recognises files by CONTENT, so names/locations don't matter:
  * creditcard  : a CSV containing a 'Class' column and V1..V28
  * PaySim      : a CSV with columns step,type,amount,...,isFraud
  * IEEE-CIS    : train_transaction.csv (+ optional train_identity.csv),
                  recognised by a 'TransactionID' column and 'isFraud'

Only needs: pandas.
"""
from __future__ import annotations
import glob, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)


def all_csvs():
    # every csv under this folder; reading our own outputs back is harmless/idempotent
    return glob.glob(os.path.join(HERE, "**", "*.csv"), recursive=True)


def header(path):
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def main():
    csvs = all_csvs()
    if not csvs:
        sys.exit("No .csv files found in this folder. Unzip the downloads here first.")

    found = {k: None for k in ("creditcard", "paysim", "ieee_tx", "ieee_id")}
    for p in csvs:
        cols = set(header(p))
        base = os.path.basename(p).lower()
        if "test" in base:
            continue  # ignore IEEE competition TEST files (unlabeled)
        if base in ("paysim.csv", "ieee_cis.csv"):
            continue  # these are OUR outputs, never sources (always regenerated)
        if {"Class", "V1", "V28"} <= cols and found["creditcard"] is None:
            found["creditcard"] = p
        elif {"step", "type", "amount", "isFraud"} <= cols and found["paysim"] is None:
            found["paysim"] = p
        elif {"TransactionID", "isFraud", "TransactionAmt"} <= cols:
            # real train_transaction (sample_submission has isFraud but NOT TransactionAmt)
            found["ieee_tx"] = p
        elif "TransactionID" in cols and "isFraud" not in cols and "identity" in base:
            # train_identity (test_identity already skipped above)
            found["ieee_id"] = p

    # --- creditcard ---
    if found["creditcard"]:
        df = pd.read_csv(found["creditcard"])
        df.to_csv(os.path.join(DATA, "creditcard.csv"), index=False)
        print(f"creditcard.csv  rows={len(df):,} fraud={int(df['Class'].sum()):,} "
              f"<- {os.path.relpath(found['creditcard'], HERE)}")
    else:
        print("creditcard: not found (ok if you already have data/creditcard.csv)")

    # --- paysim ---
    if found["paysim"]:
        df = pd.read_csv(found["paysim"])
        for c in ["isFlaggedFraud", "nameOrig", "nameDest"]:
            if c in df.columns:
                df = df.drop(columns=c)
        df.to_csv(os.path.join(DATA, "paysim.csv"), index=False)
        print(f"paysim.csv      rows={len(df):,} fraud={int(df['isFraud'].sum()):,} "
              f"<- {os.path.relpath(found['paysim'], HERE)}")
    else:
        print("paysim: not found (download ealaxi/paysim1 and unzip here)")

    # --- ieee-cis (merge transaction + identity) ---
    if found["ieee_tx"]:
        tx = pd.read_csv(found["ieee_tx"])
        if found["ieee_id"]:
            idy = pd.read_csv(found["ieee_id"])
            df = tx.merge(idy, on="TransactionID", how="left")
            src = f"{os.path.basename(found['ieee_tx'])}+{os.path.basename(found['ieee_id'])}"
        else:
            df = tx
            src = os.path.basename(found["ieee_tx"]) + " (no identity file found)"
        if "TransactionID" in df.columns:
            df = df.drop(columns=["TransactionID"])
        df.to_csv(os.path.join(DATA, "ieee_cis.csv"), index=False)
        print(f"ieee_cis.csv    rows={len(df):,} cols={df.shape[1]} "
              f"fraud={int(df['isFraud'].sum()):,} <- {src}")
    else:
        print("ieee-cis: not found (download the competition's train_transaction.csv "
              "[+ train_identity.csv] and unzip here)")

    print("\nReady files are in ./data/. Now run:")
    print("    python qedhtn_pipeline.py --config run.example.json --outdir all_tables --max-cardinality 50")


if __name__ == "__main__":
    main()
