import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

csv_path = "Most Streamed Spotify Songs 2024.csv"
df = pd.read_csv(csv_path, encoding="latin1")

df.columns = df.columns.str.strip()

print("Columns in dataset:")
print(df.columns)

print("\nBasic numeric summary (df.describe):")

if "All Time Rank" in df.columns:
    df["All Time Rank"] = pd.to_numeric(df["All Time Rank"], errors="coerce")

num_df = df.select_dtypes(include=[np.number])

print(num_df.describe())

if "All Time Rank" in num_df.columns:
    corr_with_target = num_df.corr()["All Time Rank"].sort_values()
    print("\nCorrelation with All Time Rank:")
    print(corr_with_target)

    plt.figure(figsize=(6, 4))
    num_df["All Time Rank"].dropna().hist(bins=30)
    plt.xlabel("All Time Rank")
    plt.ylabel("Count")
    plt.title("Distribution of All Time Rank")
    plt.tight_layout()
    plt.savefig("all_time_rank_hist.png", dpi=300)
    print("\nSaved CP1-style histogram: all_time_rank_hist.png")
else:
    print("\nWARNING: 'All Time Rank' not found in numeric columns when doing EDA.")
    
TARGET_COL = "All Time Rank" 

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found. Check df.columns and update TARGET_COL.")

non_feature_cols = [
    TARGET_COL,
    "Track", "Track Name", "Track name",
    "Artist", "Artist Name",
    "Album", "Album Name",
    "Release Date", "ISRC"
]

non_feature_cols = [c for c in non_feature_cols if c in df.columns]

numeric_df = df.drop(columns=non_feature_cols, errors="ignore")

numeric_df = numeric_df.select_dtypes(include=[np.number])

numeric_df = numeric_df.fillna(0)

y = pd.to_numeric(df[TARGET_COL], errors="coerce")

mask = ~y.isna()
X = numeric_df.loc[mask]
y = y.loc[mask]

print("\nFinal feature columns:")
print(list(X.columns))
print("X shape:", X.shape, "| y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain size:", X_train.shape[0], "| Test size:", X_test.shape[0])

def evaluate_model(name, model, X_train, X_test, y_train, y_test, X_full, y_full, cv_splits=5):
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    r2 = r2_score(y_test, y_test_pred)

    n = len(y_test)
    p = X_train.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, X_full, y_full, cv=kf,
        scoring="neg_root_mean_squared_error"
    )
    cv_rmse = -np.mean(cv_scores)

    print(f"\n=== {name} ===")
    print(f"Train RMSE: {train_rmse:.2f}")
    print(f"Test  RMSE: {test_rmse:.2f}")
    print(f"CV    RMSE: {cv_rmse:.2f}")
    print(f"R^2:        {r2:.4f}")
    print(f"Adj R^2:    {adj_r2:.4f}")

    return {
        "name": name,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "cv_rmse": cv_rmse,
        "r2": r2,
        "adj_r2": adj_r2,
        "y_test": y_test,
        "y_test_pred": y_test_pred
    }

variant1 = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(
        n_neighbors=5,
        weights="uniform",
        metric="euclidean"
    ))
])

variant2 = Pipeline([
    ("scaler", RobustScaler()),
    ("knn", KNeighborsRegressor(
        n_neighbors=10,
        weights="uniform",
        metric="manhattan"
    ))
])

variant3 = Pipeline([
    ("scaler", MinMaxScaler()),
    ("knn", KNeighborsRegressor(
        n_neighbors=5,
        weights="distance",
        metric="euclidean"
    ))
])

results = []

results.append(evaluate_model(
    "Variant 1: StdScaler + Euclidean (k=5, uniform)",
    variant1, X_train, X_test, y_train, y_test, X, y
))

results.append(evaluate_model(
    "Variant 2: RobustScaler + Manhattan (k=10, uniform)",
    variant2, X_train, X_test, y_train, y_test, X, y
))

results.append(evaluate_model(
    "Variant 3: MinMax + Euclidean (k=5, distance)",
    variant3, X_train, X_test, y_train, y_test, X, y
))

results_df = pd.DataFrame(results)[
    ["name", "train_rmse", "test_rmse", "cv_rmse", "r2", "adj_r2"]
]

print("\n=== SUMMARY TABLE (COPY THIS TO SLIDES) ===")
print(results_df.to_string(index=False))


import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 12
})

best = [r for r in results if "RobustScaler + Manhattan" in r["name"]][0]

y_test_best = best["y_test"]
y_pred_best = best["y_test_pred"]
residuals = y_test_best - y_pred_best

plt.figure(figsize=(6, 4))
plt.scatter(y_pred_best, residuals, alpha=0.5)
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Predicted All Time Rank")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot – KNN (RobustScaler + Manhattan, k = 10)")
plt.tight_layout()
plt.savefig("residual_plot_best_variant.png", dpi=300)

plt.figure(figsize=(6, 4))
plt.scatter(y_test_best, y_pred_best, alpha=0.5)
min_val = min(y_test_best.min(), y_pred_best.min())
max_val = max(y_test_best.max(), y_pred_best.max())
plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)
plt.xlabel("Actual All Time Rank")
plt.ylabel("Predicted All Time Rank")
plt.title("Predicted vs Actual – KNN (RobustScaler + Manhattan)")
plt.tight_layout()
plt.savefig("pred_vs_actual_best_variant.png", dpi=300)

print("\nSaved plots as:")
print("  residual_plot_best_variant.png")
print("  pred_vs_actual_best_variant.png")