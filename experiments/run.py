"""Run the mSDA + fusion + spectral-clustering pipeline on a dataset.

Examples
--------
    python experiments/run.py --dataset syn-mixed
    python experiments/run.py --dataset syn-mixed --grid
    python experiments/run.py --dataset heart --affinity rbf        # requires `ucimlrepo`
"""
import argparse
import itertools
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import spearmanr

from msda import (make_synthetic, load_uci, preprocess,
                  build_fused_similarity, MarginalizedSDA, spectral_cluster, evaluate)


def get_data(name):
    if name.startswith("syn-"):
        kind = name.split("-", 1)[1]            # syn-mixed / syn-numerical / syn-categorical
        return make_synthetic(kind=kind, n_per_class=400, n_classes=3, seed=0)
    return load_uci(name)


def run_once(X_enc, Xn, Xc, y, gamma, alpha, lam, noise, n_layers, affinity, n_neighbors):
    S = build_fused_similarity(Xn, Xc, gamma=gamma, alpha=alpha)
    Z = MarginalizedSDA(n_layers=n_layers, noise=noise, lam=lam).fit_transform(X_enc, S_fused=S)
    k = len(np.unique(y))
    y_pred = spectral_cluster(Z, n_clusters=k, affinity=affinity, n_neighbors=n_neighbors)
    return evaluate(Z, y, y_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="syn-mixed")
    ap.add_argument("--affinity", default="nearest_neighbors", choices=["rbf", "nearest_neighbors"])
    ap.add_argument("--n_neighbors", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--grid", action="store_true", help="small grid search + unsupervised selection")
    args = ap.parse_args()

    X_num, X_cat, y = get_data(args.dataset)
    X_enc, Xn, Xc = preprocess(X_num, X_cat)
    print(f"dataset={args.dataset}  n={len(y)}  classes={len(np.unique(y))}  "
          f"d_num={Xn.shape[1]}  d_onehot={Xc.shape[1]}\n")

    if not args.grid:
        m = run_once(X_enc, Xn, Xc, y, args.gamma, args.alpha, args.lam,
                     args.noise, args.n_layers, args.affinity, args.n_neighbors)
        for k, v in m.items():
            print(f"  {k:16s}: {v:.3f}")
        return

    # ---- small grid search -------------------------------------------------
    grid = list(itertools.product(
        [0.1, 0.5, 1.0],          # gamma
        [0.2, 0.5, 0.7],          # alpha
        [0.1, 1.0],               # lam
        [0.1, 0.3],               # noise
        [2, 3],                   # n_layers
    ))
    rows = []
    for g, a, l, ns, nl in grid:
        m = run_once(X_enc, Xn, Xc, y, g, a, l, ns, nl, args.affinity, args.n_neighbors)
        m.update(dict(gamma=g, alpha=a, lam=l, noise=ns, n_layers=nl))
        rows.append(m)

    best = max(rows, key=lambda r: r["NMI"])
    print(f"best-by-NMI config: gamma={best['gamma']} alpha={best['alpha']} "
          f"lam={best['lam']} noise={best['noise']} n_layers={best['n_layers']}")
    for k in ("NMI", "ARI", "ACC", "Silhouette", "CalinskiHarabasz", "DaviesBouldin"):
        if k in best:
            print(f"  {k:16s}: {best[k]:.3f}")

    # ---- unsupervised model selection: which internal metric tracks NMI? ----
    nmi = [r["NMI"] for r in rows]
    print("\nSpearman correlation of internal metrics with NMI (label-free selection proxy):")
    for internal in ("Silhouette", "CalinskiHarabasz", "DaviesBouldin"):
        vals = [r.get(internal, np.nan) for r in rows]
        if not any(np.isnan(vals)):
            rho = spearmanr(vals, nmi).correlation
            print(f"  {internal:16s}: rho = {rho:+.3f}")


if __name__ == "__main__":
    main()
