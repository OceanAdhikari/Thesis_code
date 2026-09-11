import io
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# =====================================================================
# 1. CONFIGURATION: Paths, test years, and RF results
# =====================================================================
seasons = {
    "Monsoon": {
        "path": r"C:\Users\ocean\OneDrive\Desktop\OA Thesis\Standarized Data\Monsoon\RF\Book1.csv",
        "test_year": 2023,
        "rf_r2": 0.67, "rf_rmse": 4.03, "rf_mae": 3.18, "rf_mape": 22.18,
    },
    "Post-Monsoon": {
        "path": r"C:\Users\ocean\OneDrive\Desktop\OA Thesis\Standarized Data\Postmonsoon\RF\Book1.csv",
        "test_year": 2022,
        "rf_r2": 0.78, "rf_rmse": 6.07, "rf_mae": 5.17, "rf_mape": 20.49,
    },
    "Pre-Monsoon": {
        "path": r"C:\Users\ocean\OneDrive\Desktop\OA Thesis\Standarized Data\Premonsoon\RF\Book1.csv",
        "test_year": 2023,
        "rf_r2": 0.61, "rf_rmse": 19.89, "rf_mae": 15.46, "rf_mape": 30.61,
    },
    "Winter": {
        "path": r"C:\Users\ocean\OneDrive\Desktop\OA Thesis\Standarized Data\Winter\RF\Book1.csv",
        "test_year": 2019,  # Dec 2018 + Jan 2019 + Feb 2019 (Model A)
        "rf_r2": 0.57, "rf_rmse": 18.05, "rf_mae": 13.10, "rf_mape": 20.87,
    },
}

# =====================================================================
# 2. EVALUATION FUNCTION
# =====================================================================
def evaluate_model(y_true, y_pred, label):
    """Calculate R2, RMSE, MAE, MAPE matching the RF script exactly."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    
    r2 = r2_score(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    mape = np.mean(np.abs((yt - yp) / (yt + 1e-8))) * 100
    
    return {
        "label": label,
        "r2": round(r2, 4),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "mape": round(mape, 2),
        "n": int(mask.sum()),
    }

# =====================================================================
# 3. MAIN LOOP: Process all 4 seasons
# =====================================================================
all_results = []

for season_name, cfg in seasons.items():
    
    # --- Load Data ---
    df = pd.read_csv(cfg["path"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # --- PROPER METEOROLOGICAL YEAR GROUPING ---
    if season_name == "Winter":
        # Shift December into following calendar year (Dec 2018 + Jan/Feb 2019 = 2019)
        df['season_year'] = df['date'].apply(lambda d: d.year + 1 if d.month == 12 else d.year)
    else:
        df['season_year'] = df['date'].dt.year

    # --- Calculate PERSISTENCE BASELINES FIRST (on RAW PM2.5) ---
    df["persist_1d"] = df.groupby("season_year")["PM2.5"].shift(1)
    df["persist_3d"] = df.groupby("season_year")["PM2.5"].transform(
        lambda s: s.shift(1).rolling(window=3).mean()
    )

    # --- Calculate RF Features (to ensure exact same row dropping) ---
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

    # --- Drop NAs ONLY AFTER all shifts/lags are calculated ---
    df = df.dropna().reset_index(drop=True)

    # --- Filter by Test Year ---
    test_df = df[df["season_year"] == cfg["test_year"]].copy()
    y_actual = test_df["PM2.5"].values

    print(f"Sanity Check -> {season_name} Test Set Range: {test_df['date'].min().date()} to {test_df['date'].max().date()} (N={len(test_df)})")

    # --- Evaluate Models ---
    m_persist_1d = evaluate_model(y_actual, test_df["persist_1d"].values, "Persistence (1-day lag)")
    m_persist_3d = evaluate_model(y_actual, test_df["persist_3d"].values, "Persistence (3-day rolling)")
    m_rf = {
        "label": "Random Forest (Proposed)",
        "r2": cfg["rf_r2"], "rmse": cfg["rf_rmse"], "mae": cfg["rf_mae"], "mape": cfg["rf_mape"],
        "n": m_persist_1d["n"]
    }

    all_results.append({
        "season": season_name,
        "test_year": cfg["test_year"],
        "persist_1d": m_persist_1d,
        "persist_3d": m_persist_3d,
        "rf": m_rf
    })

# =====================================================================
# 4. PRINT PUBLICATION-READY TABLE
# =====================================================================
print("\n" + "="*80)
print("COMPARATIVE RESULTS: PERSISTENCE BASELINE VS RANDOM FOREST")
print("="*80)
print(f"\n  {'Season':<14} {'Model':<28} {'R2':>8} {'RMSE':>8} {'MAE':>8} {'MAPE':>8}")
print(f"  {'-'*14} {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

for r in all_results:
    s = r["season"]
    p1 = r["persist_1d"]
    p3 = r["persist_3d"]
    rf = r["rf"]
    
    print(f"  {s:<14} {p1['label']:<28} {p1['r2']:>8.2f} {p1['rmse']:>8.2f} {p1['mae']:>8.2f} {p1['mape']:>7.2f}%")
    print(f"  {'':<14} {p3['label']:<28} {p3['r2']:>8.2f} {p3['rmse']:>8.2f} {p3['mae']:>8.2f} {p3['mape']:>7.2f}%")
    print(f"  {'':<14} {rf['label']:<28} {rf['r2']:>8.2f} {rf['rmse']:>8.2f} {rf['mae']:>8.2f} {rf['mape']:>7.2f}%")
    print("  " + "-"*76)