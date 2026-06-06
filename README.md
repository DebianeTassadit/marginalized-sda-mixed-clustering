# Marginalized Stacked Denoising Auto-Encoders for Mixed-Type Data Clustering

A graph-regularized **marginalized stacked denoising autoencoder (mSDA)** for clustering
**mixed-type tabular data** (numerical + categorical), developed during a research internship
at **Centre Borelli (CNRS)**.

The method fuses feature-type-specific similarities (RBF kernel for numerical attributes,
cosine similarity for categorical ones) into a single affinity structure that guides a
**closed-form, layer-wise denoising** encoder, smooths the codes over that graph, applies a
`tanh` nonlinearity, and clusters the result with spectral clustering — no explicit graph and
no iterative optimization required.

> **Stack:** Python · NumPy · SciPy · scikit-learn
> Full write-up: [`paper.pdf`](./paper.pdf)

---

## Method

1. **Preprocess** - standardize numerical features, one-hot encode categorical features.
2. **Multi-view similarity** - `S_num = RBF(X_num, γ)`, `S_cat = cosine(X_cat)`, row-normalized,
   then fused: `S_fused = α · S_num + (1 − α) · S_cat`.
3. **Graph-regularized mSDA** - at each layer, the dropout corruption is *analytically marginalized*
   (Chen et al., 2012), giving a closed-form map `W = (Q + λI)⁻¹ P`; the codes are smoothed over
   `S_fused` and passed through `tanh`. Layer encodings are concatenated.
4. **Spectral clustering** on the learned representation (`rbf` or `nearest_neighbors` affinity).
5. **Evaluation** - external (NMI, ARI, Accuracy via Hungarian alignment) and internal
   (Silhouette, Calinski–Harabasz, Davies–Bouldin) metrics.

## Results (NMI vs. k-SubMix, from the paper)

| Dataset | Ours (mSDA + Fusion) | k-SubMix |
|---|---|---|
| Heart | **0.353** | 0.350 |
| Adult | **0.180** | 0.173 |
| Derma | **0.870** | 0.858 |
| Bands | **0.124** | 0.054 |
| Credit | 0.310 | **0.372** |

Outperforms k-SubMix on **4 of 5** real-world UCI datasets, with a notably large margin on *Bands*.
The pipeline runs in **under a few minutes per dataset on a standard CPU**.

## Repository structure

```
marginalized-sda-mixed-clustering/
├── msda/
│   ├── similarity.py     # RBF / cosine views + row-normalization + fusion
│   ├── model.py          # MarginalizedSDA: closed-form layers + graph smoothing + tanh
│   ├── clustering.py     # spectral clustering + metrics (NMI/ARI/ACC/internal)
│   └── datasets.py       # synthetic generators + preprocessing + optional UCI loaders
├── experiments/run.py    # CLI: single run or grid search + unsupervised selection
├── examples/quickstart.py
├── paper.pdf             # full technical write-up (add your PDF here)
├── requirements.txt
└── README.md
```

## Quickstart

```bash
git clone https://github.com/DebianeTassadit/marginalized-sda-mixed-clustering.git
cd marginalized-sda-mixed-clustering
pip install -r requirements.txt

python examples/quickstart.py
# -> NMI ~0.74, ACC ~0.85 on a synthetic mixed-type dataset
```

Grid search + label-free (Spearman) model selection on a dataset:

```bash
python experiments/run.py --dataset syn-mixed --grid
```

Run on a UCI dataset (requires `pip install ucimlrepo pandas`):

```bash
python experiments/run.py --dataset heart --affinity rbf
```

## Credit

Research-internship work by **Tassadit Debiane**, supervised by **Dr. Lazhar Labiod**,
Centre Borelli, CNRS — Université Paris Cité. All datasets are open/public (UCI ML Repository
and synthetic benchmarks following the k-SubMix protocol). Builds on the marginalized denoising
autoencoder of Chen, Xu, Weinberger & Sha (ICML 2012).
