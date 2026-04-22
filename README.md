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
