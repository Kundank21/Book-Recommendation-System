# Book Recommendation System

A recommendation system that predicts what a user should read next, based on their reading history.

## Problem Statement

Given two datasets — `chapters.csv` (50K chapters across ~50K books) and `interactions.csv` (1M user–chapter interactions across ~150K users) — build a system that recommends books and chapters to users.

**Key Insight:** EDA revealed that 99% of user–book pairs involve exactly one chapter, making book-level recommendation the primary task rather than chapter sequencing.

## Approach

The system implements four recommendation strategies:

1. **Collaborative Filtering (Truncated SVD)** — Matrix factorization (k=50) on the binary user–book interaction matrix to learn latent co-reading patterns.
2. **Content-Based Filtering (TF-IDF)** — Genre tags encoded as TF-IDF vectors; user profiles built as normalised mean of read-book vectors; books ranked by cosine similarity.
3. **Adaptive Hybrid** — Weighted blend of CF and content-based scores with adaptive α: cold users (≤2 books) lean content-based (α=0.3), warm users (5+) lean CF (α=0.7).
4. **Cold-Start Fallback** — Popularity-based recommendations with optional genre filtering for new users with zero history.

A lightweight **chapter-level lookup** layer surfaces the next unread chapter once a book is recommended.

## Results

Evaluation uses leave-one-out split with 99 negative samples per test instance.

| Model | HR@5 | HR@10 | HR@20 | NDCG@10 | MRR |
|-------|------|-------|-------|---------|-----|
| Popularity | 0.1498 | 0.2070 | 0.2935 | 0.1283 | 0.1229 |
| CF (SVD) | 0.1809 | 0.2525 | 0.3543 | 0.1586 | 0.1483 |
| Content-Based | 0.1156 | 0.1771 | 0.2665 | 0.1087 | 0.0979 |
| **Hybrid** | **0.1896** | **0.2627** | **0.3653** | **0.1648** | **0.1543** |

The Hybrid model outperforms all baselines across every metric.

## Repository Structure

```
├── Untitled12.ipynb        # Main notebook (EDA, models, evaluation, examples)
├── chapters.csv            # Book/chapter metadata (place in same directory)
├── interactions.csv        # User–chapter interactions (place in same directory)
├── solution_writeup.pdf    # One-page solution write-up
└── README.md               # This file
```

## How to Run

### Requirements

- Python 3.8+
- Libraries: `pandas`, `numpy`, `scipy`, `scikit-learn`

Install dependencies:

```bash
pip install pandas numpy scipy scikit-learn
```

### Steps

1. Place `chapters.csv` and `interactions.csv` in the same directory as the notebook.
2. Open `Untitled12.ipynb` in Jupyter Notebook or JupyterLab.
3. Run all cells sequentially (Kernel → Restart & Run All).
4. The notebook will output EDA findings, model training, evaluation metrics, example recommendations, and cold-start demonstrations.

## Key Design Decisions

- **Binary interactions** — With 99% single-chapter pairs, treating read/not-read as binary loses no practical signal.
- **SVD rank 50** — Balances expressiveness vs. overfitting on a very sparse matrix (~0.01% density).
- **Adaptive α blending** — Smoothly transitions from content-based (cold users) to CF (warm users) without a hard cutoff.
- **Chapter lookup, not a sequence model** — A simple "next unread chapter" lookup is more appropriate than a sequence model given the data distribution.

## Future Improvements

- Author co-occurrence as collaborative signal
- Implicit ALS (e.g., `implicit` library) for better handling of binary feedback
- Temporal weighting — recent reads weighted higher
- Graph-based methods (LightGCN) on the user–book bipartite graph
- Sequence-aware model (SASRec) for the small multi-chapter user subset
