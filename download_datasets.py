#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_datasets.py
====================
Fetch the three public benchmarks used by the QEDHTN pipeline and write them to
./data/ with the exact filenames and label columns that run.example.json expects:

    data/creditcard.csv   label column: Class      (UCI/ULB, ULB/Kaggle)
    data/paysim.csv       label column: isFraud     (PaySim)
    data/ieee_cis.csv     label column: isFraud     (IEEE-CIS, transaction+identity merged)

All three live on Kaggle and require a (free) Kaggle account:
  * creditcardfraud  and  paysim1  are plain datasets.
  * ieee-fraud-detection is a COMPETITION: you must click "Join Competition" /
    accept its rules once on the website before the API will let you download it:
        https://www.kaggle.com/competitions/ieee-fraud-detection/rules

ONE-TIME SETUP
--------------
1. pip install kagglehub pandas            (kaggle CLI also works; see fallback)
2. Create an API token: kaggle.com -> your avatar -> Settings -> "Create New Token".
   It downloads kaggle.json. Put it at:
       Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json
       macOS/Linux: ~/.kaggle/kaggle.json   (chmod 600)
   (Or set env vars KAGGLE_USERNAME and KAGGLE_KEY.)
3. Accept the IEEE-CIS competition rules at the link above.

RUN
---
    python download_datasets.py
    # then:
    python qedhtn_pipeline.py --config run.example.json

Notes
-----
* These files are large (creditcard ~150 MB, PaySim ~470 MB, IEEE-CIS ~1 GB after
  merge). Make sure you have a few GB free.
* IEEE-CIS has very many high-cardinality categorical columns. The pipeline's
  default get_dummies encoding can blow up memory on it; see the note printed at
  the end for the recommended --max-cardinality handling.
"""
from __future__ import annotations
import os, sys, glob, shutil

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def _have(mod):
    try:
        __import__(mod); return True
    except Exception:
        return False


def _find(root, pattern):
    hits = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    if not hits:
        raise FileNotFoundError(f"{pattern} not found under {root}")
    return max(hits, key=os.path.getsize)  # the real data file is the big one


# --------------------------------------------------------------------------- #
# Download helpers (kagglehub preferred, kaggle CLI fallback)                  #
# --------------------------------------------------------------------------- #
def dl_dataset(slug: str) -> str:
    """Return the local directory a Kaggle *dataset* was downloaded to."""
    if _have("kagglehub"):
        import kagglehub
        return kagglehub.dataset_download(slug)
    # fallback: kaggle CLI
    import subprocess, tempfile
    out = tempfile.mkdtemp(prefix="kg_")
    subprocess.check_call(["kaggle", "datasets", "download", "-d", slug,
                           "-p", out, "--unzip"])
    return out


def dl_competition(slug: str) -> str:
    """Return the local directory a Kaggle *competition* was downloaded to."""
    if _have("kagglehub"):
        import kagglehub
        return kagglehub.competition_download(slug)
    import subprocess, tempfile, zipfile
    out = tempfile.mkdtemp(prefix="kgc_")
    subprocess.check_call(["kaggle", "competitions", "download", "-c", slug,
                           "-p", out])
    for z in glob.glob(os.path.join(out, "*.zip")):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
    return out


# --------------------------------------------------------------------------- #
# Per-dataset preparation                                                      #
# --------------------------------------------------------------------------- #
def prep_creditcard():
    import pandas as pd
    print("\n[1/3] UCI/ULB (mlg-ulb/creditcardfraud) ...")
    root = dl_dataset("mlg-ulb/creditcardfraud")
    src = _find(root, "creditcard.csv")
    dst = os.path.join(DATA_DIR, "creditcard.csv")
    shutil.copy(src, dst)
    df = pd.read_csv(dst)
    assert "Class" in df.columns, "expected label column 'Class'"
    print(f"      -> {dst}  rows={len(df):,} fraud={int(df['Class'].sum()):,} "
          f"({df['Class'].mean()*100:.3f}%)")


def prep_paysim():
    import pandas as pd
    print("\n[2/3] PaySim (ealaxi/paysim1) ...")
    root = dl_dataset("ealaxi/paysim1")
    src = _find(root, "*.csv")
    dst = os.path.join(DATA_DIR, "paysim.csv")
    df = pd.read_csv(src)
    assert "isFraud" in df.columns, "expected label column 'isFraud'"
    # drop leakage-prone identifier / helper columns, keep modelling features
    drop = [c for c in ["isFlaggedFraud", "nameOrig", "nameDest"] if c in df.columns]
    df = df.drop(columns=drop)
    df.to_csv(dst, index=False)
    print(f"      -> {dst}  rows={len(df):,} fraud={int(df['isFraud'].sum()):,} "
          f"({df['isFraud'].mean()*100:.3f}%)  dropped={drop}")


def prep_ieee():
    import pandas as pd
    print("\n[3/3] IEEE-CIS (competition ieee-fraud-detection) ...")
    print("      (requires accepting the competition rules once on the website)")
    root = dl_competition("ieee-fraud-detection")
    tx = _find(root, "train_transaction.csv")
    idy = _find(root, "train_identity.csv")
    dft = pd.read_csv(tx)
    dfi = pd.read_csv(idy)
    df = dft.merge(dfi, on="TransactionID", how="left")
    assert "isFraud" in df.columns, "expected label column 'isFraud'"
    df = df.drop(columns=[c for c in ["TransactionID"] if c in df.columns])
    dst = os.path.join(DATA_DIR, "ieee_cis.csv")
    df.to_csv(dst, index=False)
    print(f"      -> {dst}  rows={len(df):,} cols={df.shape[1]} "
          f"fraud={int(df['isFraud'].sum()):,} ({df['isFraud'].mean()*100:.3f}%)")


def main():
    if not (_have("kagglehub") or shutil.which("kaggle")):
        sys.exit("Install kagglehub (pip install kagglehub) or the kaggle CLI first. "
                 "See the setup notes at the top of this file.")
    try:
        prep_creditcard()
        prep_paysim()
        prep_ieee()
    except Exception as e:
        print("\nERROR:", e)
        print("Common causes: kaggle.json not placed / no token; IEEE-CIS rules "
              "not accepted yet; or network. See the setup notes at the top.")
        sys.exit(1)

    print("\nDone. All three CSVs are in ./data/. Now run:")
    print("    python qedhtn_pipeline.py --config run.example.json")
    print("\nIEEE-CIS note: it has hundreds of high-cardinality categorical columns. "
          "If one-hot encoding runs out of memory, cap or hash high-cardinality "
          "columns before modelling (e.g. keep the top-K categories per column, or "
          "drop columns with >1000 unique values), or restrict to the numeric + "
          "low-cardinality columns. Keep whatever choice you make identical across "
          "the Q-FARM / RBF / MI runs so the comparison stays fair.")


if __name__ == "__main__":
    main()
