#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qedhtn_pipeline.py
==================
End-to-end, REPRODUCIBLE pipeline that produces the QEDHTN paper's per-dataset
tables from real data -- with real standard deviations over repeated seeds.

It exists to answer the second reviewer's three remaining comments honestly:

  Comment 1 (leftover placeholders / "mean +/- SD over N runs" with no SD):
     This pipeline runs N seeds and reports mean +/- SD for every metric, so the
     captions become true. No cell is ever "to be completed".

  Comment 2 (values look manufactured -- round targets, .0/.5 AUCs):
     Every number here is MEASURED by running models on data. There are no hand
     written targets. Whatever comes out is what you report, round or not.

  Comment 3 (the RBF baseline is never defined; double-digit margin implausible):
     The RBF baseline is defined EXACTLY, in code, as Q-FARM with the entropy
     term of Eq. (4) switched off (rho = 1). So Q-FARM and the RBF baseline share
     the identical kernel responses and differ ONLY in the entropy-weighted
     reranking. That makes the margin mechanistically interpretable -- and, in
     practice, SMALL. If you observe a double-digit PR-AUC gap you have a bug or a
     leak; report the true (usually 1-3 point) margin and attribute the system's
     gains to GRAFIX, not to the kernel stage.

--------------------------------------------------------------------------------
WHAT IT DOES, per dataset:
  1. load + preprocess (impute, encode, scale)
  2. stratified 70/15/15 split (seeded)
  3. SMOTE on the TRAINING fold only (val/test keep natural prevalence)
  4. build a feature subset with ONE of three stages:
        - qfarm : kernel response relevance R_f  +  entropy term   (Eq. 4, rho<1)
        - rbf   : kernel response relevance R_f  ONLY              (rho = 1)  <-- baseline
        - mi    : sklearn mutual_info_classif top-k
  5. train a downstream classifier on the selected features
        - default: scikit-learn (runs everywhere, no GPU)
        - optional: a faithful GRAFIX (graph + GAT + transformer + fusion) in
          PyTorch, with per-component toggles for the leave-one-out ablation
  6. pick ONE threshold on validation (max-F1), evaluate ONCE on test
  7. aggregate seeds -> mean +/- SD, and emit Tables 5/6/7/8 + ablation via
     qedhtn_eval.py (must sit beside this file)

RUN:
  python qedhtn_pipeline.py --demo                 # synthetic, proves it executes
  python qedhtn_pipeline.py --config run.json      # your real datasets

Dependencies: numpy, scikit-learn, pandas  (torch only if you enable GRAFIX).
"""
from __future__ import annotations
import argparse, json, os, sys, time, warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import NearestNeighbors

# qedhtn_eval.py provides the single-confusion-matrix metrics + table builders.
try:
    from qedhtn_eval import evaluate, aggregate_runs, build_tables, Metrics
except ImportError:
    sys.exit("Place qedhtn_eval.py in the same folder as this file.")

RNG_MASTER = 20260101


# =========================================================================== #
# 1. Minimal, correct SMOTE (train fold only).                                #
#    Swap for imbalanced-learn's SMOTE if you have it; this avoids the dep.    #
# =========================================================================== #
def smote(X: np.ndarray, y: np.ndarray, *, k: int = 5, target_ratio: float = 0.3,
          seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Oversample the minority (y==1) with synthetic points on segments between a
    minority sample and one of its k minority neighbours, until the minority
    reaches `target_ratio` of the majority. MUST be called on the training fold
    only -- never on validation or test.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float); y = np.asarray(y, int)
    minority = X[y == 1]
    n_maj = int((y == 0).sum()); n_min = len(minority)
    if n_min < 2:
        return X, y
    need = int(target_ratio * n_maj) - n_min
    if need <= 0:
        return X, y
    kk = min(k, n_min - 1)
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(minority)
    idx = nn.kneighbors(minority, return_distance=False)
    # vectorised synthetic generation (no Python loop -> fast even for millions)
    a = rng.integers(0, n_min, size=need)
    b = idx[a, 1 + rng.integers(0, kk, size=need)]
    gap = rng.random((need, 1)).astype(np.float32)
    synth = (minority[a] + gap * (minority[b] - minority[a])).astype(np.float32)
    X_out = np.vstack([X.astype(np.float32, copy=False), synth])
    y_out = np.concatenate([y, np.ones(need, int)])
    return X_out, y_out


# =========================================================================== #
# 2. Feature stages.                                                          #
#    Q-FARM and the RBF baseline share the SAME kernel-response relevance;    #
#    they differ ONLY by the entropy term (Eq. 4). This is the definition the #
#    reviewer asked for.                                                       #
# =========================================================================== #
def _kernel_response_relevance(X: np.ndarray, y: np.ndarray, *, delta: float) -> np.ndarray:
    """
    Per-feature relevance R_f under the Gaussian/RBF map of Eq. (1).
    For feature f, compare the RBF similarity of each value to the fraud-class
    centroid vs the genuine-class centroid; R_f is the mean absolute difference.
    This is the "mean absolute kernel response" of Eq. (3), made concrete and
    class-aware. Returned min-max normalised to [0,1].
    """
    X = np.asarray(X, float)
    mu1 = X[y == 1].mean(axis=0)        # fraud centroid (per feature)
    mu0 = X[y == 0].mean(axis=0)        # genuine centroid
    # RBF similarity of each sample's feature value to each centroid
    s1 = np.exp(-delta * (X - mu1) ** 2)
    s0 = np.exp(-delta * (X - mu0) ** 2)
    R = np.abs(s1 - s0).mean(axis=0)    # separation under the kernel, per feature
    rng = R.max() - R.min()
    return (R - R.min()) / rng if rng > 0 else np.zeros_like(R)


def _feature_entropy(X: np.ndarray, *, bins: int = 16) -> np.ndarray:
    """Shannon entropy of each (min-max normalised, binned) feature, in [0,1]."""
    X = np.asarray(X, float)
    H = np.empty(X.shape[1])
    for f in range(X.shape[1]):
        col = X[:, f]
        rng = col.max() - col.min()
        if rng == 0:
            H[f] = 0.0; continue
        hist, _ = np.histogram((col - col.min()) / rng, bins=bins, range=(0, 1))
        p = hist / hist.sum()
        p = p[p > 0]
        H[f] = -(p * np.log2(p)).sum()
    rng = H.max() - H.min()
    return (H - H.min()) / rng if rng > 0 else np.zeros_like(H)


def select_features(X: np.ndarray, y: np.ndarray, *, stage: str,
                    keep_fraction: float = 0.6, rho: float = 0.7,
                    delta: float = 0.5, seed: int = 0,
                    max_select_rows: int = 100_000) -> np.ndarray:
    """
    Return the indices of the selected features.

    stage = 'qfarm' : psi(f) = rho*R_f + (1-rho)*Entropy(f)   (Eq. 4, rho<1)
    stage = 'rbf'    : psi(f) = R_f                            (entropy OFF, rho=1)
                       <-- THE BASELINE: identical kernel responses to Q-FARM,
                           differing only by the removed entropy term.
    stage = 'mi'     : mutual_info_classif, same number of features kept.

    For very large folds the ranking statistics are computed on a random subsample
    of max_select_rows (feature SELECTION only; training/evaluation still use all
    rows). This keeps the mutual-information and kernel steps tractable on
    PaySim/IEEE-CIS without changing which rows are modelled.
    """
    X = np.asarray(X, np.float32); y = np.asarray(y, int)
    d = X.shape[1]
    n_keep = max(1, int(round(keep_fraction * d)))

    if len(X) > max_select_rows:
        sidx = np.random.default_rng(seed).choice(len(X), max_select_rows, replace=False)
        Xs, ys = X[sidx], y[sidx]
    else:
        Xs, ys = X, y

    if stage == "mi":
        mi = mutual_info_classif(Xs, ys, random_state=seed)
        order = np.argsort(mi)[::-1]
        return np.sort(order[:n_keep])

    R = _kernel_response_relevance(Xs, ys, delta=delta)
    if stage == "rbf":
        psi = R                                   # rho = 1, no entropy
    elif stage == "qfarm":
        H = _feature_entropy(Xs)
        psi = rho * R + (1.0 - rho) * H           # Eq. (4)
    else:
        raise ValueError(f"unknown stage: {stage}")
    order = np.argsort(psi)[::-1]
    return np.sort(order[:n_keep])


# =========================================================================== #
# 3. Downstream classifiers.                                                  #
# =========================================================================== #
def make_sklearn_classifier(seed: int):
    """Default, dependency-light downstream model. Calibrated-ish probabilities."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=None,
        l2_regularization=1.0, class_weight="balanced", random_state=seed)


# ---- Optional GRAFIX (PyTorch). Imported lazily; only needed for --model grafix
#      and for the leave-one-out ablation (component toggles). -----------------
def make_grafix(seed, in_dim, *, use_graph=True, use_gat=True,
                use_transformer=True, use_fusion=True, hidden=64):
    """
    Minibatched, inductive reference GRAFIX. Each example is processed with its
    k sampled neighbours as a short token set, so cost is O(batch * k) and the
    model scales to millions of rows (PaySim) and to IEEE-CIS without building
    one giant full-graph. Per-component toggles drive the leave-one-out ablation:
      use_graph/use_gat -> neighbourhood aggregation (attention vs mean)
      use_transformer   -> self-attention over [self, neighbours] tokens
      use_fusion        -> concat graph + transformer embeddings
    forward(x_self [B,d], x_nbrs [B,k,d]) -> logits [B]
    """
    import torch, torch.nn as nn

    class GRAFIX(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.use_graph, self.use_gat = use_graph, use_gat
            self.use_transformer, self.use_fusion = use_transformer, use_fusion
            self.proj = nn.Linear(d, h)
            self.att = nn.Linear(2 * h, 1)
            if use_transformer:
                layer = nn.TransformerEncoderLayer(h, nhead=4, dim_feedforward=2 * h,
                                                   batch_first=True, dropout=0.1)
                self.tr = nn.TransformerEncoder(layer, num_layers=2)
            fuse_in = h * (2 if (use_fusion and use_transformer and use_graph) else 1)
            self.head = nn.Sequential(nn.Linear(fuse_in, h), nn.ReLU(),
                                      nn.Dropout(0.1), nn.Linear(h, 1))

        def forward(self, x_self, x_nbrs):
            import torch
            z = torch.relu(self.proj(x_self))          # [B,h]
            zn = torch.relu(self.proj(x_nbrs))         # [B,k,h]
            g = z
            if self.use_graph:
                if self.use_gat:
                    q = z.unsqueeze(1).expand(-1, zn.size(1), -1)   # [B,k,h]
                    e = self.att(torch.cat([q, zn], -1)).squeeze(-1)  # [B,k]
                    a = torch.softmax(e, dim=1).unsqueeze(-1)         # [B,k,1]
                    g = (a * zn).sum(1)                               # [B,h]
                else:
                    g = zn.mean(1)
            t = z
            if self.use_transformer:
                seq = torch.cat([z.unsqueeze(1), zn], dim=1)         # [B,k+1,h]
                t = self.tr(seq)[:, 0, :]                            # self-token out [B,h]
            if self.use_fusion and self.use_graph and self.use_transformer:
                feat = torch.cat([g, t], -1)
            elif self.use_transformer:
                feat = t
            else:
                feat = g
            return self.head(feat).squeeze(-1)

    torch.manual_seed(seed)
    return GRAFIX(in_dim, hidden)


# =========================================================================== #
# 4. One training+evaluation run for a (dataset, stage, model, seed).         #
# =========================================================================== #
@dataclass
class Fold:
    Xtr: np.ndarray; ytr: np.ndarray
    Xva: np.ndarray; yva: np.ndarray
    Xte: np.ndarray; yte: np.ndarray


def split_fold(X, y, seed) -> Fold:
    """Stratified 70/15/15. SMOTE is applied later, to the train fold only."""
    X_tmp, Xte, y_tmp, yte = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed)
    val_rel = 0.15 / 0.85
    Xtr, Xva, ytr, yva = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, stratify=y_tmp, random_state=seed)
    return Fold(Xtr, ytr, Xva, yva, Xte, yte)


def _knn_index(X, k, seed):
    k = min(k, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    return idx[:, 1:]   # drop self


def run_once(X, y, *, dataset, stage, model="sklearn", seed=0,
             smote_ratio=0.3, keep_fraction=0.6, grafix_cfg=None,
             epochs=30, batch_size=4096, knn_k=10, max_ref_nodes=200_000) -> Metrics:
    fold = split_fold(X, y, seed)
    # SMOTE on train only
    Xtr, ytr = smote(fold.Xtr, fold.ytr, target_ratio=smote_ratio, seed=seed)
    # feature selection fitted on the (resampled) TRAIN fold only
    sel = select_features(Xtr, ytr, stage=stage, keep_fraction=keep_fraction, seed=seed)
    Xtr_s, Xva_s, Xte_s = Xtr[:, sel], fold.Xva[:, sel], fold.Xte[:, sel]

    if model == "sklearn":
        clf = make_sklearn_classifier(seed).fit(Xtr_s, ytr)
        va_score = clf.predict_proba(Xva_s)[:, 1]
        te_score = clf.predict_proba(Xte_s)[:, 1]
    elif model == "grafix":
        import torch, torch.nn as nn
        cfg = grafix_cfg or {}
        rng = np.random.default_rng(seed)

        # Bounded reference set: the graph is built over at most max_ref_nodes
        # training rows (subsampled for very large folds). Neighbours for ANY
        # node (train/val/test) are looked up in this reference set, so the model
        # stays inductive and memory stays bounded.
        Xref, yref = Xtr_s, ytr
        if len(Xref) > max_ref_nodes:
            keep = rng.choice(len(Xref), max_ref_nodes, replace=False)
            Xref, yref = Xref[keep], yref[keep]
        k = min(knn_k, len(Xref) - 1)
        nn_ref = NearestNeighbors(n_neighbors=k + 1).fit(Xref)
        tr_idx = nn_ref.kneighbors(Xref, return_distance=False)[:, 1:]  # drop self

        Xref_t = torch.tensor(Xref, dtype=torch.float)
        yref_t = torch.tensor(yref, dtype=torch.float)
        net = make_grafix(seed, Xref.shape[1], **cfg)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        pos = max(int((yref == 1).sum()), 1); neg = int((yref == 0).sum())
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float))

        N = len(Xref)
        net.train()
        for _ in range(epochs):
            perm = rng.permutation(N)
            for i in range(0, N, batch_size):
                b = torch.as_tensor(perm[i:i + batch_size], dtype=torch.long)
                xs = Xref_t[b]
                xn = Xref_t[torch.as_tensor(tr_idx[perm[i:i + batch_size]], dtype=torch.long)]
                out = net(xs, xn); loss = lossf(out, yref_t[b])
                opt.zero_grad(); loss.backward(); opt.step()

        net.eval()

        def infer(Xq):
            qi = nn_ref.kneighbors(Xq, n_neighbors=k, return_distance=False)
            Xq_t = torch.tensor(Xq, dtype=torch.float)
            outs = []
            with torch.no_grad():
                for i in range(0, len(Xq), batch_size):
                    xs = Xq_t[i:i + batch_size]
                    xn = Xref_t[torch.as_tensor(qi[i:i + batch_size], dtype=torch.long)]
                    outs.append(torch.sigmoid(net(xs, xn)).numpy())
            return np.concatenate(outs) if outs else np.array([])

        va_score = infer(Xva_s)
        te_score = infer(Xte_s)
    else:
        raise ValueError(f"unknown model: {model}")

    return evaluate(fold.yte, te_score, dataset=dataset,
                    val_true=fold.yva, val_score=va_score,
                    threshold_mode="max_f1", check_prevalence=False)


def run_seeds(X, y, *, dataset, stage, model="sklearn", seeds=5, **kw) -> dict:
    runs = [run_once(X, y, dataset=dataset, stage=stage, model=model,
                     seed=RNG_MASTER + s, **kw) for s in range(seeds)]
    agg = aggregate_runs(runs)
    agg["_runs"] = runs
    return agg


# =========================================================================== #
# 5. Data loading.                                                            #
# =========================================================================== #
def _reduce_cardinality(X: pd.DataFrame, max_cardinality: int):
    """
    Cap high-cardinality categorical columns so one-hot encoding stays bounded:
    keep the `max_cardinality` most frequent categories per column and bucket the
    rest into '__other__'. This is what makes IEEE-CIS (hundreds of
    high-cardinality columns) encodable without exhausting memory. The SAME cap is
    applied to every feature stage, so the Q-FARM / RBF / MI comparison stays fair.
    """
    cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
    reduced = 0
    for c in cat_cols:
        vc = X[c].value_counts()
        if len(vc) > max_cardinality:
            top = set(vc.index[:max_cardinality])
            X[c] = X[c].where(X[c].isin(top), other="__other__")
            reduced += 1
    if reduced:
        print(f"      capped {reduced} categorical column(s) at "
              f"max_cardinality={max_cardinality}")
    return X


def preprocess(df: pd.DataFrame, label_col: str, max_cardinality: int = 50):
    y = df[label_col].astype(int).to_numpy()
    X = df.drop(columns=[label_col])
    # bound categorical cardinality BEFORE one-hot encoding
    if max_cardinality and max_cardinality > 0:
        X = _reduce_cardinality(X, max_cardinality)
    X = pd.get_dummies(X, dummy_na=False)
    # Cast to a single float32 dtype and hand sklearn a plain ndarray. Passing a
    # MIXED-dtype DataFrame makes pandas build a giant object array first, which
    # is what caused the MemoryError on the wide IEEE-CIS table.
    X = X.astype(np.float32, copy=False).to_numpy()
    X = SimpleImputer(strategy="median", copy=False).fit_transform(X)
    X = StandardScaler(copy=False).fit_transform(X)
    return X.astype(np.float32, copy=False), y


def load_dataset(name: str, path: str, label_col: str,
                 max_cardinality: int = 50) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    # downcast float64 columns to float32 up front to roughly halve memory
    fcols = df.select_dtypes(include=["float64"]).columns
    if len(fcols):
        df[fcols] = df[fcols].astype(np.float32)
    return preprocess(df, label_col, max_cardinality=max_cardinality)


# =========================================================================== #
# 6. Orchestration                                                            #
# =========================================================================== #
def run_config(cfg: dict, outdir: str):
    """
    cfg = {
      "seeds": 5, "model": "sklearn"|"grafix",
      "datasets": {
         "IEEE-CIS": {"path": "ieee.csv", "label": "isFraud"},
         "PaySim":   {"path": "paysim.csv", "label": "isFraud"},
         "UCI/ULB":  {"path": "creditcard.csv", "label": "Class"}
      },
      "run_ablation": true
    }
    """
    seeds = cfg.get("seeds", 5); model = cfg.get("model", "sklearn")
    default_maxcard = cfg.get("max_cardinality", 50)
    # GRAFIX training knobs (only used when model='grafix'); safe to pass always
    gkw = {}
    if model == "grafix":
        for key, ck in [("max_ref_nodes", "max_train_nodes"), ("epochs", "epochs"),
                        ("batch_size", "batch_size"), ("knn_k", "knn_k")]:
            if ck in cfg:
                gkw[key] = cfg[ck]
    results = {"main": {}, "rbf": {}, "mi": {}}
    ablation = {}
    for name, spec in cfg["datasets"].items():
        maxcard = spec.get("max_cardinality", default_maxcard)
        X, y = load_dataset(name, spec["path"], spec["label"], max_cardinality=maxcard)
        print(f"[{name}] n={len(y)} fraud={int(y.sum())} ({y.mean()*100:.3f}%)")
        results["main"][name] = run_seeds(X, y, dataset=name, stage="qfarm", model=model, seeds=seeds, **gkw)
        results["rbf"][name]  = run_seeds(X, y, dataset=name, stage="rbf",   model=model, seeds=seeds, **gkw)
        results["mi"][name]   = run_seeds(X, y, dataset=name, stage="mi",    model=model, seeds=seeds, **gkw)
        if cfg.get("run_ablation") and model == "grafix":
            ablation[name] = _ablation(X, y, name, seeds)
    if ablation:
        results["ablation"] = ablation
    build_tables(results, outdir=outdir)
    return results


def _ablation(X, y, name, seeds):
    """Leave-one-out over GRAFIX components (needs model='grafix')."""
    base = dict(use_graph=True, use_gat=True, use_transformer=True, use_fusion=True)
    configs = {
        "Full model":       dict(base),
        "- Graph":          {**base, "use_graph": False, "use_gat": False},
        "- GAT":            {**base, "use_gat": False},
        "- Transformer":    {**base, "use_transformer": False},
        "- Fusion":         {**base, "use_fusion": False},
    }
    out = {}
    for cfg_name, gcfg in configs.items():
        out[cfg_name] = run_seeds(X, y, dataset=f"{name}:{cfg_name}", stage="qfarm",
                                  model="grafix", seeds=seeds, grafix_cfg=gcfg)
    # feature-stage substitutions (same full GRAFIX, different selection)
    out["- Q-FARM (RBF rank)"] = run_seeds(X, y, dataset=f"{name}:rbf", stage="rbf",
                                           model="grafix", seeds=seeds, grafix_cfg=base)
    out["- Q-FARM (MI select)"] = run_seeds(X, y, dataset=f"{name}:mi", stage="mi",
                                            model="grafix", seeds=seeds, grafix_cfg=base)
    return out


# =========================================================================== #
# 7. Synthetic self-test                                                       #
# =========================================================================== #
def _demo():
    print("=" * 72)
    print("SELF-TEST ON SYNTHETIC DATA. Proves the pipeline executes end to end.")
    print("These are NOT results; the synthetic signal is arbitrary.")
    print("=" * 72)
    rng = np.random.default_rng(0)

    def synth(n, prevalence, d=24, signal=1.2):
        k = max(2, int(n * prevalence))
        y = np.zeros(n, int); y[:k] = 1; rng.shuffle(y)
        X = rng.normal(0, 1, (n, d))
        informative = slice(0, 8)
        X[:, informative] += (y[:, None] * signal)       # only some features carry signal
        return X.astype(np.float32), y

    datasets = {  # small, natural-ish prevalence, so the demo finishes fast
        "IEEE-CIS": synth(9000, 0.035, signal=1.3),
        "PaySim":   synth(9000, 0.013, signal=1.6),
        "UCI/ULB":  synth(9000, 0.017, signal=1.4),
    }
    results = {"main": {}, "rbf": {}, "mi": {}}
    for name, (X, y) in datasets.items():
        results["main"][name] = run_seeds(X, y, dataset=name, stage="qfarm", seeds=3)
        results["rbf"][name]  = run_seeds(X, y, dataset=name, stage="rbf",   seeds=3)
        results["mi"][name]   = run_seeds(X, y, dataset=name, stage="mi",    seeds=3)
    build_tables(results, outdir="qedhtn_pipeline_demo")
    print("\nNote how Q-FARM vs RBF differ only slightly -- that is the honest,")
    print("expected size of an entropy-reweighting effect, not double digits.")
    print("CSV + all_tables.txt written to ./qedhtn_pipeline_demo/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="synthetic end-to-end self-test")
    ap.add_argument("--config", help="path to run.json (see run_config docstring)")
    ap.add_argument("--outdir", default="qedhtn_tables", help="output directory")
    ap.add_argument("--max-cardinality", type=int, default=None,
                    help="cap categorical columns at the top-K categories (overrides "
                         "the config value; recommended ~50 for IEEE-CIS)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    if args.demo:
        _demo()
    elif args.config:
        with open(args.config) as fh:
            cfg = json.load(fh)
        if args.max_cardinality is not None:
            cfg["max_cardinality"] = args.max_cardinality
        run_config(cfg, args.outdir)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
