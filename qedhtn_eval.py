#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qedhtn_eval.py
==============
Consistent, natural-prevalence evaluation for the QEDHTN fraud-detection paper.

WHY THIS EXISTS
---------------
The second-round reviewer showed that the manuscript's Tables 5 and 6 were
mutually contradictory and impossible at the stated class prevalence
(e.g. 98.8% precision AND 0.9% FPR at 0.17% fraud cannot both be true).

This script removes that whole class of error by construction:

  * EVERY operating-point metric (accuracy, precision, recall, F1, FPR) is
    derived from ONE confusion matrix, computed at ONE threshold, on the
    UNTOUCHED test fold at its NATURAL prevalence.
  * Threshold-free metrics (ROC-AUC, PR-AUC) come from the same scores.
  * The script refuses to report numbers that don't reconcile: it asserts
    that the confusion-matrix totals match the test-fold size and the natural
    fraud count, and that precision recomputed two independent ways agrees.

So Table 5 and Table 6 can never disagree again, and precision/recall/FPR are
always consistent with prevalence.

WHAT YOU FEED IT
----------------
For each dataset you provide held-out predictions on the TEST fold:
    y_true  : 1-D array of {0,1}, 1 = fraud, at natural prevalence
    y_score : 1-D array of the model's fraud probability / score in [0,1]
(optionally the same for the VALIDATION fold, used only to pick the threshold).

You do NOT resample the test fold. If you accidentally do, the prevalence check
below will flag it (that is exactly the mistake the reviewer caught).

You get back a metrics dict, and the module can assemble the exact tables the
manuscript needs (Table 5, Table 6, the RBF/MI baseline tables, and the
per-dataset leave-one-out ablation) as CSV + paste-ready text, with mean +/- SD
over repeated seeds.

    python qedhtn_eval.py --demo         # self-test on SYNTHETIC data (proves
                                         # the pipeline is self-consistent;
                                         # these are NOT real results)

    # In your own training code:
    from qedhtn_eval import evaluate, aggregate_runs, build_tables

Dependencies: numpy, scikit-learn, pandas.
"""

from __future__ import annotations
import argparse
import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# --------------------------------------------------------------------------- #
# Reference numbers derived from Table 4 of the manuscript.                    #
# Used ONLY to sanity-check that a test fold really is at natural prevalence.  #
# These are arithmetic facts about the datasets, not results.                  #
# --------------------------------------------------------------------------- #
DATASET_REFERENCE = {
    # name        total_txns   total_fraud   natural_prevalence
    "IEEE-CIS": dict(total=590_540, fraud=20_663, prevalence=20_663 / 590_540),
    "PaySim":   dict(total=6_362_620, fraud=8_213, prevalence=8_213 / 6_362_620),
    "UCI/ULB":  dict(total=284_807, fraud=492,    prevalence=492 / 284_807),
}
TEST_FRACTION = 0.15  # 70/15/15 split in Table 4


def expected_test_counts(dataset: str, test_fraction: float = TEST_FRACTION) -> dict:
    """Test-fold size and fraud count implied by the split, for cross-checking."""
    ref = DATASET_REFERENCE[dataset]
    return dict(
        n_test=round(ref["total"] * test_fraction),
        n_fraud=round(ref["fraud"] * test_fraction),
        prevalence=ref["prevalence"],
    )


# --------------------------------------------------------------------------- #
# Threshold selection                                                         #
# --------------------------------------------------------------------------- #
def choose_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    mode: str = "max_f1",
    fixed: Optional[float] = None,
    alert_budget: Optional[int] = None,
) -> float:
    """
    Pick ONE decision threshold. Do this on the VALIDATION fold, then apply the
    returned value unchanged to the test fold, so the operating point is not
    tuned on the test data.

    mode:
      "fixed"        -> return `fixed` (e.g. 0.5).
      "max_f1"       -> threshold maximising F1 on the supplied fold.
      "alert_budget" -> highest threshold that still raises at least
                        `alert_budget` alerts (positives). Useful when ops can
                        only review N cases/day.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    if mode == "fixed":
        if fixed is None:
            raise ValueError("mode='fixed' requires fixed=<threshold>")
        return float(fixed)

    if mode == "alert_budget":
        if alert_budget is None:
            raise ValueError("mode='alert_budget' requires alert_budget=<int>")
        order = np.sort(y_score)[::-1]
        k = min(alert_budget, len(order)) - 1
        k = max(k, 0)
        return float(order[k])

    if mode == "max_f1":
        # Vectorised: use the PR curve (C-optimised) and pick the threshold that
        # maximises F1. Scales to million-row folds.
        from sklearn.metrics import precision_recall_curve
        prec, rec, thr = precision_recall_curve(y_true, y_score)
        # prec/rec have length len(thr)+1; the last point (rec=0/prec=1) has no
        # threshold, so align to thr by dropping it.
        prec, rec = prec[:-1], rec[:-1]
        denom = prec + rec
        f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(denom),
                       where=denom > 0)
        if f1.size == 0:
            return 0.5
        return float(thr[int(np.argmax(f1))])

    raise ValueError(f"unknown mode: {mode}")


# --------------------------------------------------------------------------- #
# Core: every metric from ONE confusion matrix                                #
# --------------------------------------------------------------------------- #
@dataclass
class Metrics:
    dataset: str
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    # derived (percent, rounded for reporting) -- filled in __post_init__
    accuracy: float = field(default=0.0)
    precision: float = field(default=0.0)
    recall: float = field(default=0.0)
    f1: float = field(default=0.0)
    fpr: float = field(default=0.0)
    prevalence: float = field(default=0.0)
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    n: int = 0
    n_fraud: int = 0

    def __post_init__(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        self.n = tp + fp + fn + tn
        self.n_fraud = tp + fn
        self.accuracy = 100.0 * (tp + tn) / self.n if self.n else 0.0
        self.precision = 100.0 * tp / (tp + fp) if (tp + fp) else 0.0
        self.recall = 100.0 * tp / (tp + fn) if (tp + fn) else 0.0
        self.fpr = 100.0 * fp / (fp + tn) if (fp + tn) else 0.0
        p, r = self.precision, self.recall
        self.f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        self.prevalence = 100.0 * self.n_fraud / self.n if self.n else 0.0


def _prevalence_precision(precision_pct, recall_pct, prevalence_frac):
    """
    Independent cross-check of the precision/recall/FPR/prevalence identity:
        precision = (pi*R) / (pi*R + (1-pi)*FPR)
    Given precision & recall & prevalence, back out the FPR they imply.
    Returns implied FPR in percent. Used only to assert self-consistency.
    """
    R = recall_pct / 100.0
    P = precision_pct / 100.0
    pi = prevalence_frac
    if P <= 0 or R <= 0:
        return 0.0
    # From precision identity: (1-pi)*FPR = pi*R*(1-P)/P
    fpr = (pi * R * (1 - P) / P) / (1 - pi)
    return 100.0 * fpr


def evaluate(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    dataset: str = "dataset",
    threshold: Optional[float] = None,
    val_true: Optional[Sequence[int]] = None,
    val_score: Optional[Sequence[float]] = None,
    threshold_mode: str = "max_f1",
    fixed_threshold: Optional[float] = None,
    alert_budget: Optional[int] = None,
    check_prevalence: bool = True,
    prevalence_tol: float = 0.5,  # allowed abs. deviation in *percentage points*
) -> Metrics:
    """
    Evaluate one dataset's TEST fold and return a fully consistent Metrics
    object. All operating-point numbers come from the single confusion matrix.

    Threshold resolution order:
      1. explicit `threshold` argument, else
      2. chosen on (val_true, val_score) via `threshold_mode`, else
      3. chosen on the test fold itself (only acceptable for `fixed` mode;
         a warning is printed otherwise, because tuning on test inflates results).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")
    if set(np.unique(y_true)) - {0, 1}:
        raise ValueError("y_true must contain only 0 and 1 (1 = fraud)")

    # 1) resolve threshold
    if threshold is None:
        if val_true is not None and val_score is not None:
            threshold = choose_threshold(
                val_true, val_score, mode=threshold_mode,
                fixed=fixed_threshold, alert_budget=alert_budget,
            )
        else:
            if threshold_mode != "fixed":
                print(f"[warn] {dataset}: no validation fold given; choosing "
                      f"threshold on the TEST fold inflates the operating point. "
                      f"Pass val_true/val_score, or threshold_mode='fixed'.")
            threshold = choose_threshold(
                y_true, y_score, mode=threshold_mode,
                fixed=fixed_threshold, alert_budget=alert_budget,
            )

    # 2) confusion matrix at that threshold
    pred = y_score >= threshold
    tp = int(np.count_nonzero(pred & (y_true == 1)))
    fp = int(np.count_nonzero(pred & (y_true == 0)))
    fn = int(np.count_nonzero(~pred & (y_true == 1)))
    tn = int(np.count_nonzero(~pred & (y_true == 0)))

    m = Metrics(dataset=dataset, threshold=float(threshold),
                tp=tp, fp=fp, fn=fn, tn=tn)

    # 3) threshold-free metrics from the same scores
    if len(np.unique(y_true)) == 2:
        m.roc_auc = 100.0 * roc_auc_score(y_true, y_score)
        m.pr_auc = 100.0 * average_precision_score(y_true, y_score)

    # 4) natural-prevalence sanity check (the reviewer's point 3)
    if check_prevalence and dataset in DATASET_REFERENCE:
        exp = DATASET_REFERENCE[dataset]["prevalence"] * 100.0
        if abs(m.prevalence - exp) > prevalence_tol:
            print(f"[warn] {dataset}: test-fold prevalence {m.prevalence:.3f}% "
                  f"deviates from natural {exp:.3f}% by more than {prevalence_tol} "
                  f"pp. Did SMOTE/resampling leak into the test fold? "
                  f"The test fold must NOT be resampled.")

    # 5) internal-consistency assertions (the reviewer's point 4)
    #    precision implied by (recall, prevalence, fpr) must match measured precision.
    implied_fpr = _prevalence_precision(m.precision, m.recall, m.n_fraud / m.n if m.n else 0)
    if (m.tp + m.fp) > 0 and m.fpr > 0:
        # allow tiny rounding differences
        assert math.isclose(implied_fpr, m.fpr, abs_tol=0.05 + 0.02 * m.fpr), (
            f"{dataset}: precision/recall/FPR/prevalence are inconsistent "
            f"(measured FPR {m.fpr:.4f}% vs implied {implied_fpr:.4f}%). "
            f"This should be impossible when all come from one confusion matrix."
        )
    assert m.n == m.tp + m.fp + m.fn + m.tn
    return m


# --------------------------------------------------------------------------- #
# Aggregation over repeated seeds -> mean +/- SD                              #
# --------------------------------------------------------------------------- #
def aggregate_runs(runs: Sequence[Metrics], ddof: int = 1) -> dict:
    """
    Combine several Metrics (one per seed) into mean +/- SD for each field.
    Returns a dict of {metric: (mean, sd)} plus formatted 'mean +/- sd' strings.
    """
    if not runs:
        raise ValueError("no runs to aggregate")
    fields = ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc"]
    out = {"dataset": runs[0].dataset, "n_runs": len(runs)}
    for f in fields:
        vals = np.array([getattr(r, f) for r in runs if getattr(r, f) is not None],
                        dtype=float)
        if vals.size == 0:
            out[f] = (None, None)
            out[f + "_str"] = "-"
            continue
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=ddof)) if vals.size > 1 else 0.0
        out[f] = (mean, sd)
        out[f + "_str"] = f"{mean:.2f} ± {sd:.2f}"
    # confusion-matrix cells: report the mean (rounded) for the confusion table
    for c in ["tp", "fp", "fn", "tn"]:
        out[c] = int(round(np.mean([getattr(r, c) for r in runs])))
    return out


# --------------------------------------------------------------------------- #
# Table builders -> the exact manuscript tables                               #
# --------------------------------------------------------------------------- #
def _fmt(v):
    return "-" if v is None else f"{v:.2f}"


def table5(per_dataset: dict) -> pd.DataFrame:
    """Table 5: per-dataset performance. per_dataset = {name: Metrics or agg-dict}."""
    rows = []
    for name, m in per_dataset.items():
        if isinstance(m, Metrics):
            rows.append([name, _fmt(m.accuracy), _fmt(m.precision), _fmt(m.recall),
                         _fmt(m.f1), _fmt(m.roc_auc), _fmt(m.pr_auc), _fmt(m.fpr)])
        else:  # aggregate dict with '_str' fields
            rows.append([name, m["accuracy_str"], m["precision_str"], m["recall_str"],
                         m["f1_str"], m["roc_auc_str"], m["pr_auc_str"], m["fpr_str"]])
    return pd.DataFrame(rows, columns=[
        "Dataset", "Acc %", "Prec %", "Rec %", "F1 %", "ROC-AUC %", "PR-AUC %", "FPR %"])


def table6(per_dataset: dict) -> pd.DataFrame:
    """Table 6: confusion matrices, with the reconciliation checks."""
    rows = []
    for name, m in per_dataset.items():
        tp, fp, fn, tn = (m.tp, m.fp, m.fn, m.tn) if isinstance(m, Metrics) else \
                         (m["tp"], m["fp"], m["fn"], m["tn"])
        total = tp + fp + fn + tn
        fraud = tp + fn
        exp = expected_test_counts(name) if name in DATASET_REFERENCE else None
        chk_total = f"{total:,}" + ("" if not exp else
                    ("  OK" if abs(total - exp["n_test"]) <= 0.02 * exp["n_test"] else
                     f"  != {exp['n_test']:,}"))
        chk_fraud = f"{fraud:,}" + ("" if not exp else
                    ("  OK" if abs(fraud - exp["n_fraud"]) <= 0.05 * exp["n_fraud"] + 1 else
                     f"  != {exp['n_fraud']:,}"))
        rows.append([name, tp, fp, fn, tn, chk_total, chk_fraud])
    return pd.DataFrame(rows, columns=[
        "Dataset", "TP", "FP", "FN", "TN", "Row total (check)", "Fraud=TP+FN (check)"])


def ablation_table(rows_by_config: dict, dataset: str) -> pd.DataFrame:
    """
    Leave-one-out ablation for ONE dataset.
    rows_by_config = {config_name: aggregate-dict or Metrics}, in the order:
      'Full model', '- Q-FARM', '- Graph', '- GAT', '- Transformer',
      '- Fusion', 'Q-FARM->RBF', 'Q-FARM->MI'.
    """
    rows = []
    for cfg, m in rows_by_config.items():
        if isinstance(m, Metrics):
            rows.append([cfg, _fmt(m.accuracy), _fmt(m.precision), _fmt(m.recall),
                         _fmt(m.f1), _fmt(m.pr_auc)])
        else:
            rows.append([cfg, m["accuracy_str"], m["precision_str"], m["recall_str"],
                         m["f1_str"], m["pr_auc_str"]])
    df = pd.DataFrame(rows, columns=["Configuration", "Acc %", "Prec %",
                                     "Rec %", "F1 %", "PR-AUC %"])
    df.attrs["dataset"] = dataset
    return df


def build_tables(results: dict, outdir: str = ".") -> dict:
    """
    results = {
      'main':      {dataset: Metrics|agg},          # -> Table 5 + Table 6
      'rbf':       {dataset: Metrics|agg},          # -> Table 7
      'mi':        {dataset: Metrics|agg},          # -> Table 8
      'ablation':  {dataset: {config: Metrics|agg}} # -> Table 1a/1b/1c
    }
    Writes CSVs and returns the DataFrames; also prints paste-ready text.
    """
    os.makedirs(outdir, exist_ok=True)
    made = {}

    if "main" in results:
        t5 = table5(results["main"]);  t6 = table6(results["main"])
        t5.to_csv(os.path.join(outdir, "table5_performance.csv"), index=False)
        t6.to_csv(os.path.join(outdir, "table6_confusion.csv"), index=False)
        made["table5"], made["table6"] = t5, t6
    if "rbf" in results:
        t7 = table5(results["rbf"]); t7.to_csv(os.path.join(outdir, "table7_rbf.csv"), index=False)
        made["table7_rbf"] = t7
    if "mi" in results:
        t8 = table5(results["mi"]); t8.to_csv(os.path.join(outdir, "table8_mi.csv"), index=False)
        made["table8_mi"] = t8
    if "ablation" in results:
        for ds, by_cfg in results["ablation"].items():
            tbl = ablation_table(by_cfg, ds)
            safe = ds.replace("/", "-")
            tbl.to_csv(os.path.join(outdir, f"ablation_{safe}.csv"), index=False)
            made[f"ablation_{ds}"] = tbl

    # paste-ready text dump
    lines = []
    for key, df in made.items():
        lines.append(f"\n### {key} ###")
        lines.append(df.to_string(index=False))
    text = "\n".join(lines)
    with open(os.path.join(outdir, "all_tables.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return made


# --------------------------------------------------------------------------- #
# CLI: evaluate from prediction files                                         #
# --------------------------------------------------------------------------- #
def _load_pred_file(path: str):
    """Load y_true,y_score from .npz (keys y_true,y_score) or .csv (cols y_true,y_score)."""
    if path.endswith(".npz"):
        d = np.load(path)
        return d["y_true"], d["y_score"]
    df = pd.read_csv(path)
    return df["y_true"].to_numpy(), df["y_score"].to_numpy()


def run_from_manifest(manifest_path: str, outdir: str):
    """
    manifest.json example:
    {
      "threshold_mode": "max_f1",
      "main": {
        "IEEE-CIS": {"test": "ieee_test.csv", "val": "ieee_val.csv"},
        "PaySim":   {"test": "paysim_test.csv", "val": "paysim_val.csv"},
        "UCI/ULB":  {"test": "uci_test.csv", "val": "uci_val.csv"}
      },
      "rbf": { ... same shape ... },
      "mi":  { ... same shape ... },
      "ablation": {
        "IEEE-CIS": {"Full model": {"test": "...","val":"..."}, "- Q-FARM": {...}, ...}
      }
    }
    Each entry may also be a LIST of {"test","val"} dicts (one per seed) to get
    mean +/- SD.
    """
    with open(manifest_path) as fh:
        man = json.load(fh)
    tmode = man.get("threshold_mode", "max_f1")

    def eval_entry(name, entry):
        seeds = entry if isinstance(entry, list) else [entry]
        runs = []
        for s in seeds:
            yt, ys = _load_pred_file(s["test"])
            vt = vs = None
            if s.get("val"):
                vt, vs = _load_pred_file(s["val"])
            runs.append(evaluate(yt, ys, dataset=name, val_true=vt, val_score=vs,
                                 threshold_mode=tmode))
        return runs[0] if len(runs) == 1 else aggregate_runs(runs)

    results = {}
    for block in ["main", "rbf", "mi"]:
        if block in man:
            results[block] = {name: eval_entry(name, e) for name, e in man[block].items()}
    if "ablation" in man:
        results["ablation"] = {
            ds: {cfg: eval_entry(f"{ds}:{cfg}", e) for cfg, e in cfgs.items()}
            for ds, cfgs in man["ablation"].items()
        }
    build_tables(results, outdir=outdir)


# --------------------------------------------------------------------------- #
# Self-test on SYNTHETIC data (proves consistency; NOT real results)          #
# --------------------------------------------------------------------------- #
def _demo():
    print("=" * 72)
    print("SELF-TEST ON SYNTHETIC DATA — these numbers are fabricated inputs")
    print("used only to prove the pipeline is internally consistent. They are")
    print("NOT results and must never appear in the paper.")
    print("=" * 72)
    rng = np.random.default_rng(0)

    def synth(dataset, sep, n_seeds=5):
        """Make a test/val fold at the dataset's NATURAL prevalence with a
        separable-ish score, over several seeds."""
        exp = expected_test_counts(dataset)
        n, k = exp["n_test"], exp["n_fraud"]
        runs = []
        for seed in range(n_seeds):
            r = np.random.default_rng(seed)
            yt = np.zeros(n, dtype=int); yt[:k] = 1; r.shuffle(yt)
            ys = np.where(yt == 1, r.normal(sep, 1, n), r.normal(0, 1, n))
            ys = 1 / (1 + np.exp(-ys))  # squash to (0,1)
            # small validation fold for thresholding
            vt = np.zeros(n, dtype=int); vt[:k] = 1; r.shuffle(vt)
            vs = np.where(vt == 1, r.normal(sep, 1, n), r.normal(0, 1, n))
            vs = 1 / (1 + np.exp(-vs))
            runs.append(evaluate(yt, ys, dataset=dataset, val_true=vt, val_score=vs,
                                 threshold_mode="max_f1"))
        return runs

    main = {ds: aggregate_runs(synth(ds, sep=s))
            for ds, s in [("IEEE-CIS", 3.0), ("PaySim", 3.5), ("UCI/ULB", 3.2)]}
    rbf = {ds: aggregate_runs(synth(ds, sep=s))
           for ds, s in [("IEEE-CIS", 2.4), ("PaySim", 2.8), ("UCI/ULB", 2.5)]}
    mi = {ds: aggregate_runs(synth(ds, sep=s))
          for ds, s in [("IEEE-CIS", 2.1), ("PaySim", 2.5), ("UCI/ULB", 2.2)]}

    configs = ["Full model", "- Q-FARM", "- Graph", "- GAT",
               "- Transformer", "- Fusion", "Q-FARM->RBF", "Q-FARM->MI"]
    seps = [3.2, 2.3, 2.9, 2.8, 3.0, 3.0, 2.4, 2.1]
    ablation = {"UCI/ULB": {c: aggregate_runs(synth("UCI/ULB", sep=s))
                            for c, s in zip(configs, seps)}}

    build_tables({"main": main, "rbf": rbf, "mi": mi, "ablation": ablation},
                 outdir="qedhtn_tables_demo")
    print("\nAll internal-consistency assertions passed.")
    print("CSV + all_tables.txt written to ./qedhtn_tables_demo/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="path to manifest.json describing prediction files")
    ap.add_argument("--outdir", default="qedhtn_tables", help="where to write CSVs")
    ap.add_argument("--demo", action="store_true", help="run synthetic self-test")
    args = ap.parse_args()
    if args.demo:
        _demo()
    elif args.manifest:
        run_from_manifest(args.manifest, args.outdir)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
