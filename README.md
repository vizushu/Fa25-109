# COGS 109 Final Project – Spotify KNN Model

This repo contains the code for my COGS 109 final project, using the  
**Most Streamed Spotify Songs 2024** dataset and a set of KNN regression models
to predict **All Time Rank**.

The main script includes:
- Basic EDA (summary stats, correlations, histogram of target)
- Feature selection and train/test split
- Three KNN variants with different scalers and distance metrics
- Evaluation using Train/Test RMSE, 5-fold CV RMSE, R², and Adjusted R²
- Residual plot and Predicted vs Actual plot for the best model

---

## Files

- `cogs109_knn_renier.py` – main script with EDA + KNN models
- `all_time_rank_hist.png` – histogram of **All Time Rank**
- `residual_plot_best_variant.png` – residual plot for best KNN variant
- `pred_vs_actual_best_variant.png` – predicted vs actual plot for best KNN variant
- `requirements.txt` – Python dependencies
- `.gitignore` – ignore venv, cache, etc.
- `README.md` – this file

> **Dataset note:**  
> The script expects the dataset file to be named:
> `Most Streamed Spotify Songs 2024.csv`  
> and placed in the same folder as the script.

You can download the dataset from Kaggle:  
**Most Streamed Spotify Songs 2024** and save it with that exact filename.

---

## How to run

### 1. Clone the repo

```bash
git clone <THIS-REPO-URL>.git
cd <THIS-REPO-NAME>