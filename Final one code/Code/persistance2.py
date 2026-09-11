import io
import os
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# Suppress warnings and fix console encoding
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Set global font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# =====================================================================
# CONFIGURATION
# =====================================================================
output_dir = r"C:\Users\ocean\OneDrive\Desktop\OA Thesis\Standarized Data\persistance\Time_Series_Plots"
os.makedirs(output_dir, exist_ok=True)

seasons_config = {
    "Monsoon": {
        "path": r"C:\Users\ocean\OneDrive\Desktop\OA Thesis\Standarized Data\Monsoon\RF\Book1.csv",
        "test_year": 2023,
    },
    "Post-Monsoon": {
        "path": r"C:\Users\ocean\OneDrive\Desktop\OA Thesis\Standarized Data\Postmonsoon\RF\Book1.csv",
        "test_year": 2022,
    },
    "Pre-Monsoon": {
        "path": r"C:\Users\ocean\OneDrive\Desktop\OA Thesis\Standarized Data\Premonsoon\RF\Book1.csv",
        "test_year": 2023,
    },
    "Winter": {
        "path": r"C:\Users\ocean\OneDrive\Desktop\OA Thesis\Standarized Data\Winter\RF\Book1.csv",
        "test_year": 2019,  # Dec 2018 + Jan/Feb 2019
    },
}

param_grid = {
    "n_estimators": [300, 500],
    "max_depth": [4, 5, 6],
    "min_samples_split": [15, 20],
    "min_samples_leaf": [8, 12],
    "max_features": ["sqrt", 0.5],
    "max_leaf_nodes": [30, 45],
    "min_impurity_decrease": [0.02, 0.05],
    "max_samples": [0.75, 0.85],
    "bootstrap": [True],
}

print("=" * 70)
print("GENERATING 3-LINE TIME SERIES PLOTS (HIGH-CONTRAST COLORS)")
print("=" * 70)

# =====================================================================
# MAIN LOOP FOR ALL 4 SEASONS
# =====================================================================
for season_name, cfg in seasons_config.items():
    print(f"\nProcessing {season_name}...")

    # 1. Load Data
    df = pd.read_csv(cfg["path"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 2. Season-Year Grouping (Solves Winter cross-year issue)
    if season_name == "Winter":
        df["season_year"] = df["date"].dt.year + (df["date"].dt.month == 12).astype(int)
    else:
        df["season_year"] = df["date"].dt.year

    # 3. Compute Persistence 1-Day Baseline (Raw PM2.5)
    df["persist_1d"] = df.groupby("season_year")["PM2.5"].shift(1)

    # 4. Feature Engineering (Capped PM2.5)
    pm25_cap = df["PM2.5"].quantile(0.97)
    df["PM2.5_feat"] = df["PM2.5"].clip(upper=pm25_cap)

    for lag in [1, 2]:
        df[f"PM2.5_lag_{lag}d"] = df.groupby("season_year")["PM2.5_feat"].shift(lag)
    df["PM2.5_rolling_3d"] = df.groupby("season_year")["PM2.5_feat"].transform(
        lambda s: s.shift(1).rolling(window=3).mean()
    )
    df["PM2.5_rolling_7d"] = df.groupby("season_year")["PM2.5_feat"].transform(
        lambda s: s.shift(1).rolling(window=7).mean()
    )

    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "Humidity", "Precipitation", "Temperature", "Wind Speed",
        "PM2.5_lag_1d", "PM2.5_lag_2d", "PM2.5_rolling_3d", "PM2.5_rolling_7d",
    ]

    # 5. Train/Test Split
    test_yr = cfg["test_year"]
    train_mask = df["season_year"] != test_yr
    test_mask = df["season_year"] == test_yr

    X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "PM2.5"]
    X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, "PM2.5"]
    p1_test = df.loc[test_mask, "persist_1d"].values
    dates_test = df.loc[test_mask, "date"].reset_index(drop=True)
    years_train_series = df.loc[train_mask, "season_year"].reset_index(drop=True)

    # 6. CV Folds
    cv_folds = []
    unique_train_years = sorted(years_train_series.unique())
    for i in range(1, len(unique_train_years)):
        val_year = unique_train_years[i]
        train_yrs = unique_train_years[:i]
        train_idx = np.where(years_train_series.isin(train_yrs))[0]
        val_idx = np.where(years_train_series == val_year)[0]
        if len(train_idx) > 0 and len(val_idx) > 0:
            cv_folds.append((train_idx, val_idx))

    # 7. Grid Search & Model Selection (Model A vs Model B)
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=cv_folds,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results["gap"] = (cv_results["mean_train_score"] - cv_results["mean_test_score"]).abs()
    top20pct = cv_results[cv_results["mean_test_score"] >= cv_results["mean_test_score"].quantile(0.80)]
    best_bal_params = cv_results.loc[top20pct["gap"].idxmin(), "params"]

    def evaluate_rf(params):
        clean = {k: v for k, v in params.items() if k != "n_jobs"}
        model = RandomForestRegressor(**clean, random_state=42)
        model.fit(X_train, y_train)
        y_tr_pred = model.predict(X_train)
        y_te_pred = model.predict(X_test)
        return y_te_pred, abs(r2_score(y_train, y_tr_pred) - r2_score(y_test, y_te_pred)), r2_score(y_test, y_te_pred)

    y_pred_A, gap_A, r2_A = evaluate_rf(grid_search.best_params_)
    y_pred_B, gap_B, r2_B = evaluate_rf(best_bal_params)

    # Select best model (using absolute gap)
    if gap_B < gap_A and r2_B >= r2_A - 0.03:
        y_test_pred = y_pred_B
    else:
        y_test_pred = y_pred_A

    # =====================================================================
    # 8. GENERATE THE 3-LINE PLOT — NEW HIGH-CONTRAST PALETTE
    # =====================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    test_indices = np.arange(len(y_test))

     # Line 1: Actual PM2.5 — Dark
    ax.plot(
        test_indices, y_test.values, "o-",
        label=r"Actual PM$_{2.5}$",
        linewidth=2.2, markersize=6, color="#000000", alpha=0.90, zorder=3
    )

    # Line 2: Random Forest Prediction — Medium
    ax.plot(
        test_indices, y_test_pred, "s-",
        label=r"Random Forest Prediction",
        linewidth=2.2, markersize=6, color="#0072B2", alpha=0.90, zorder=2
    )

    # Line 3: Persistence Baseline — Light (warm orange)
    ax.plot(
        test_indices, p1_test, "^--",
        label=r"Persistence Baseline (1-day lag)",
        linewidth=1.8, markersize=6, color="#E69F00", alpha=0.85, zorder=1
    )

    # Aesthetics
    ax.set_xlabel("Test Sample Index", fontsize=20, fontweight="bold")
    ax.set_ylabel(r"PM$_{2.5}$ ($\mu$g/m$^3$)", fontsize=20, fontweight="bold")
    ax.tick_params(axis="both", labelsize=16)

    # Legend
    ax.legend(fontsize=15, loc="best", frameon=True, facecolor="white",
              edgecolor="gray", framealpha=0.95)

    # Lighter grid so lines stand out
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save
    save_path = os.path.join(output_dir, f"{season_name}_3line_timeseries.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  -> Saved: {season_name}_3line_timeseries.png")

print("\n" + "=" * 70)
print("[SUCCESS] All 4 pictures generated with new high-contrast colors!")
print(f"Saved in: {output_dir}")