# DBSCAN From Scratch

A pure Python implementation of the **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) algorithm — no scikit-learn, no ML libraries.

## 🧠 What is DBSCAN?

DBSCAN is a density-based clustering algorithm that:
- Groups closely packed points into clusters
- Marks low-density points as **noise/outliers**
- Detects **non-spherical** cluster shapes (e.g., spirals)

## ✨ Features

- Built from scratch using only `numpy` and `matplotlib`
- Recursive cluster expansion
- Handles complex, non-linear cluster shapes
- Visual output with color-coded clusters and noise points

## 📊 Dataset

Synthetic dataset (500 samples) combining:
- Random scattered points
- Normal distribution cluster
- Spiral pattern (sine/cosine based)

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy matplotlib
```
### Run

```bash
python dbscan_from_scratch.py
```
## ⚙️ Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `eps` | `0.2` | Neighborhood search radius |
| `min_samples` | `5` | Minimum points to form a dense region |

## 🔍 How It Works

1. **`find_neighbors(X, eps, point)`** — finds all points within radius `eps` using Euclidean distance
2. **`expand_cluster(...)`** — recursively expands a cluster by visiting neighbors
3. **`DBSCAN(X, eps, min_samples)`** — main function; iterates all points and assigns cluster labels or marks as noise (`-1`)

## 📈 Output

A scatter plot where:
- Each cluster is shown in a **distinct color**
- **Black points** = noise/outliers

## 🎓 Purpose

Educational implementation to understand the inner workings of density-based clustering — ideal for ML/data science students.

## 📁 Project Structure


DBSCAN-From-Scratch/
└── dbscan_from_scratch.py

## 👤 Author

Morteza Pazhoum — @MortezaPZ

K.N. Toosi University of Technology — Computer Science
