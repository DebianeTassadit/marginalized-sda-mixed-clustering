"""Minimal end-to-end example on a synthetic mixed-type dataset.

Run:  python examples/quickstart.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msda import (make_synthetic, preprocess, build_fused_similarity,
                  MarginalizedSDA, spectral_cluster, evaluate)


def main():
    # 1. data: 3 classes, 3 numerical + 3 categorical features
    X_num, X_cat, y = make_synthetic(kind="mixed", n_per_class=400, n_classes=3, seed=0)

    # 2. preprocess -> encoded input + per-view blocks for similarity
    X_enc, X_num_s, X_cat_oh = preprocess(X_num, X_cat)

    # 3. multi-view fused similarity graph
    S = build_fused_similarity(X_num_s, X_cat_oh, gamma=0.1, alpha=0.7)

    # 4. graph-regularized marginalized SDA encoding
    Z = MarginalizedSDA(n_layers=3, noise=0.1, lam=0.1).fit_transform(X_enc, S_fused=S)

    # 5. spectral clustering + evaluation
    y_pred = spectral_cluster(Z, n_clusters=3, affinity="nearest_neighbors", n_neighbors=15)
    metrics = evaluate(Z, y, y_pred)

    print("Synthetic (mixed) — clustering metrics:")
    for k, v in metrics.items():
        print(f"  {k:16s}: {v:.3f}")


if __name__ == "__main__":
    main()
