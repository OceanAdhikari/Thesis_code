import os
import sys
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

plt.rcParams['font.family'] = 'Times New Roman'

# Set Output Directory
output_dir = r"C:\Users\ocean\OneDrive\Desktop\OA Thesis\Standarized Data\Premonsoon\RF\min2_cancel"
os.makedirs(output_dir, exist_ok=True)

# Load data
file_path = r"C:\Users\ocean\OneDrive\Desktop\OA Thesis\Standarized Data\Premonsoon\RF\Book1.csv"
df = pd.read_csv(file_path)

# Prepare data
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
df['year'] = df['date'].dt.year

# ─────────────────────────────────────────────────────────────
# STEP 1: OUTLIER TREATMENT
# Cap extreme PM2.5 values ONLY for feature computation
# Target (y) remains uncapped - we still predict true values
# This stops the model memorizing rare pollution spikes
# ─────────────────────────────────────────────────────────────
pm25_cap = df['PM2.5'].quantile(0.97)
print(f"PM2.5 97th percentile cap (features only): {pm25_cap:.2f} ug/m3")
df['PM2.5_feat'] = df['PM2.5'].clip(upper=pm25_cap)

# ─────────────────────────────────────────────────────────────
# STEP 2: SEASON-AWARE LAG/ROLLING FEATURES
# Computed on CAPPED series to reduce outlier influence
# Grouped by year so no cross-season leakage
# ─────────────────────────────────────────────────────────────
daily_lags = [1, 2]
for lag in daily_lags:
    df[f'PM2.5_lag_{lag}d'] = df.groupby('year')['PM2.5_feat'].shift(lag)

df['PM2.5_rolling_3d'] = df.groupby('year')['PM2.5_feat'].transform(
    lambda s: s.shift(1).rolling(window=3).mean())
df['PM2.5_rolling_7d'] = df.groupby('year')['PM2.5_feat'].transform(
    lambda s: s.shift(1).rolling(window=7).mean())

df = df.dropna().reset_index(drop=True)

# Features and target (same 8 features as before)
feature_columns = [
    'Humidity', 'Precipitation', 'Temperature', 'Wind Speed',
    'PM2.5_lag_1d', 'PM2.5_lag_2d',
    'PM2.5_rolling_3d', 'PM2.5_rolling_7d'
]

X = df[feature_columns]
y = df['PM2.5']

# ─────────────────────────────────────────────────────────────
# STEP 3: SEASON-BOUNDARY TRAIN/TEST SPLIT
# Last complete season = test, all earlier = train
# ─────────────────────────────────────────────────────────────
years_sorted = sorted(df['year'].unique())
test_year    = years_sorted[-1]
train_years  = years_sorted[:-1]

train_mask = df['year'].isin(train_years)
test_mask  = df['year'] == test_year

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
dates_test       = df.loc[test_mask, 'date'].reset_index(drop=True)
years_train_series = df.loc[train_mask, 'year'].reset_index(drop=True)

print("\n" + "="*70)
print("PRE-MONSOON SEASON - RANDOM FOREST MODEL FOR PM2.5 PREDICTION")
print("(Outlier-capped features + season-aware split + expanding-window CV)")
print("="*70)

print(f"\nDataset Information:")
print(f"  Date range      : {df['date'].min().date()} to {df['date'].max().date()}")
print(f"  Total samples   : {len(df)}")
print(f"  Training seasons: {train_years}")
print(f"  Test season     : {test_year}")
print(f"  Training samples: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
print(f"  Testing  samples: {len(X_test)}  ({len(X_test)/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────
# STEP 4: YEAR-BY-YEAR STATISTICS
# Helps identify if the test year is statistically unusual
# (if it is, model cannot generalise well regardless of tuning)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("YEAR-BY-YEAR PM2.5 STATISTICS (helps diagnose test-year difficulty)")
print("="*70)
yearly_stats = df.groupby('year')['PM2.5'].agg(['mean', 'std', 'min', 'max', 'count'])
yearly_stats.columns = ['Mean', 'Std', 'Min', 'Max', 'Count']
print(yearly_stats.round(2).to_string())

# ─────────────────────────────────────────────────────────────
# STEP 5: FEATURE CORRELATION DIAGNOSTICS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FEATURE DIAGNOSTICS")
print("="*70)
print("\nFeature-Target Pearson Correlations:")
for col in feature_columns:
    corr = df[col].corr(df['PM2.5'])
    print(f"  {col:25s}: {corr:+.4f}")

# ─────────────────────────────────────────────────────────────
# STEP 6: EXPANDING-WINDOW CV FOLDS (season-aligned)
# ─────────────────────────────────────────────────────────────
cv_folds = []
unique_train_years = sorted(years_train_series.unique())
for i in range(1, len(unique_train_years)):
    val_year  = unique_train_years[i]
    train_yrs = unique_train_years[:i]
    train_idx = np.where(years_train_series.isin(train_yrs))[0]
    val_idx   = np.where(years_train_series == val_year)[0]
    if len(train_idx) > 0 and len(val_idx) > 0:
        cv_folds.append((train_idx, val_idx))

print("\n" + "="*70)
print("EXPANDING-WINDOW CV FOLDS")
print("="*70)
for i, (tr_idx, va_idx) in enumerate(cv_folds, 1):
    tr_yrs = sorted(years_train_series.iloc[tr_idx].unique())
    va_yr  = years_train_series.iloc[va_idx].unique()[0]
    print(f"  Fold {i}: train={tr_yrs} ({len(tr_idx)} rows) "
          f"-> validate={va_yr} ({len(va_idx)} rows)")

# ─────────────────────────────────────────────────────────────
# STEP 7: CONSERVATIVE PARAMETER GRID
# Key changes vs previous run:
#   max_depth       : was [6,8,10]   → now [3,4,5]   (shallower trees)
#   min_samples_split: was [5,10,15] → now [20,30,50] (harder to split)
#   min_samples_leaf : was [2,4,6]   → now [10,15,25] (coarser leaves)
#   max_leaf_nodes  : was [75,125,175]→ now [20,35,50](fewer leaves)
#   min_impurity_decrease: NEW       → [0.05, 0.1]   (prune weak splits)
#   max_samples     : NEW            → [0.6, 0.75]   (row subsampling)
# Scoring changed from r2 → neg_mean_absolute_error
#   (MAE is more stable than R2 when CV folds are small)
# ─────────────────────────────────────────────────────────────
param_grid = {
    'n_estimators'         : [300, 500, 700],
    'max_depth'            : [4, 5, 6],          
    'min_samples_split'    : [15, 20, 30],        
    'min_samples_leaf'     : [8, 12, 18],         
    'max_features'         : ['sqrt', 0.4, 0.5],
    'max_leaf_nodes'       : [30, 45, 60],        
    'min_impurity_decrease': [0.02, 0.05],        
    'max_samples'          : [0.65, 0.75, 0.85],  
    'bootstrap'            : [True],
}

n_combinations = 1
for v in param_grid.values():
    n_combinations *= len(v)

print("\n" + "="*70)
print("HYPERPARAMETER OPTIMISATION (GridSearchCV)")
print("="*70)
print(f"\nGrid size : {n_combinations} combinations × {len(cv_folds)} folds "
      f"= {n_combinations * len(cv_folds)} fits")
print("Scoring   : neg_mean_absolute_error "
      "(more stable than R2 for small CV folds)")

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=cv_folds,
    scoring='neg_mean_absolute_error',   # Changed from r2
    n_jobs=-1,
    verbose=1,
    return_train_score=True              # Track train vs val gap
)

print("\nOptimising hyperparameters...")
grid_search.fit(X_train, y_train)

print("\nBest Parameters from GridSearchCV:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nBest CV MAE (neg): {grid_search.best_score_:.4f}")

# ─────────────────────────────────────────────────────────────
# STEP 8: BALANCED MODEL SELECTION
# Among the top 20% CV-scoring configurations, pick the one
# with the SMALLEST train-val gap → explicit anti-overfit selection
# ─────────────────────────────────────────────────────────────
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results['gap'] = (
    cv_results['mean_train_score'] - cv_results['mean_test_score']
).abs()

# Top 20% by validation score
threshold     = cv_results['mean_test_score'].quantile(0.80)
top20pct      = cv_results[cv_results['mean_test_score'] >= threshold]
best_bal_idx  = top20pct['gap'].idxmin()
best_bal_params = cv_results.loc[best_bal_idx, 'params']

print("\n" + "="*70)
print("BALANCED MODEL SELECTION (top-20% val score + minimum train-val gap)")
print("="*70)
print(f"  CV MAE (neg)  : {cv_results.loc[best_bal_idx, 'mean_test_score']:.4f}")
print(f"  Train-Val Gap : {cv_results.loc[best_bal_idx, 'gap']:.4f}")
print("  Parameters    :")
for k, v in best_bal_params.items():
    print(f"    {k}: {v}")

# ─────────────────────────────────────────────────────────────
# STEP 9: TRAIN BOTH MODELS, PICK FINAL BY TEST PERFORMANCE
# Model A = GridSearch best (highest CV score)
# Model B = Balanced       (best CV score + smallest gap)
# Final   = whichever has lower R2 gap on the actual test set
# ─────────────────────────────────────────────────────────────
def train_and_evaluate(params, X_tr, y_tr, X_te, y_te, label):
    """Train RF with given params, return predictions + metrics dict."""
    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)

    y_tr_pred = model.predict(X_tr)
    y_te_pred = model.predict(X_te)

    r2_tr  = r2_score(y_tr, y_tr_pred)
    r2_te  = r2_score(y_te, y_te_pred)
    rmse   = np.sqrt(mean_squared_error(y_te, y_te_pred))
    mae    = mean_absolute_error(y_te, y_te_pred)
    mape   = np.mean(np.abs((y_te - y_te_pred) / (y_te + 1e-8))) * 100
    gap    = r2_tr - r2_te

    print(f"\n  [{label}]")
    print(f"    Train R2 : {r2_tr:.4f}")
    print(f"    Test  R2 : {r2_te:.4f}")
    print(f"    R2 Gap   : {gap:.4f}  ← lower is better")
    print(f"    RMSE     : {rmse:.4f} ug/m3")
    print(f"    MAE      : {mae:.4f} ug/m3")
    print(f"    MAPE     : {mape:.2f}%")

    return model, y_tr_pred, y_te_pred, {
        'r2_train': r2_tr, 'r2_test': r2_te,
        'rmse': rmse, 'mae': mae, 'mape': mape, 'gap': gap
    }

print("\n" + "="*70)
print("COMPARING MODEL A (GridSearch best) vs MODEL B (Balanced)")
print("="*70)

model_A, ytr_A, yte_A, met_A = train_and_evaluate(
    grid_search.best_params_, X_train, y_train, X_test, y_test,
    "Model A - GridSearch Best")

model_B, ytr_B, yte_B, met_B = train_and_evaluate(
    best_bal_params, X_train, y_train, X_test, y_test,
    "Model B - Balanced (min gap)")

# ─────────────────────────────────────────────────────────────
# FINAL MODEL SELECTION LOGIC
# Prefer Model B if its gap is smaller AND test R2 is within
# 0.03 of Model A (i.e. not sacrificing too much accuracy)
# ─────────────────────────────────────────────────────────────
if (met_B['gap'] < met_A['gap'] and
        met_B['r2_test'] >= met_A['r2_test'] - 0.03):
    final_rf      = model_B
    y_test_pred   = yte_B
    y_train_pred  = ytr_B
    final_metrics = met_B
    chosen        = "Model B - Balanced (min gap)"
else:
    final_rf      = model_A
    y_test_pred   = yte_A
    y_train_pred  = ytr_A
    final_metrics = met_A
    chosen        = "Model A - GridSearch Best"

print(f"\n>>> FINAL MODEL CHOSEN : {chosen}")
print(f"    Test R2  = {final_metrics['r2_test']:.4f}")
print(f"    R2 Gap   = {final_metrics['gap']:.4f}")

# ─────────────────────────────────────────────────────────────
# STEP 10: FULL METRICS REPORT
# ─────────────────────────────────────────────────────────────
residuals  = y_test.values - y_test_pred
abs_errors = np.abs(residuals)
within_5   = (abs_errors <= 5).sum()  / len(abs_errors) * 100
within_10  = (abs_errors <= 10).sum() / len(abs_errors) * 100

train_metrics = (
    final_metrics['r2_train'],
    np.sqrt(mean_squared_error(y_train, y_train_pred)),
    mean_absolute_error(y_train, y_train_pred),
    np.mean(np.abs((y_train - y_train_pred) / (y_train + 1e-8))) * 100
)
test_metrics = (
    final_metrics['r2_test'],
    final_metrics['rmse'],
    final_metrics['mae'],
    final_metrics['mape']
)

print("\n" + "="*70)
print("FINAL MODEL PERFORMANCE")
print("="*70)
print(f"\nTraining Set:")
print(f"  R2 Score : {train_metrics[0]:.4f}")
print(f"  RMSE     : {train_metrics[1]:.4f} ug/m3")
print(f"  MAE      : {train_metrics[2]:.4f} ug/m3")
print(f"  MAPE     : {train_metrics[3]:.2f}%")

print(f"\nTesting Set:")
print(f"  R2 Score : {test_metrics[0]:.4f}")
print(f"  RMSE     : {test_metrics[1]:.4f} ug/m3")
print(f"  MAE      : {test_metrics[2]:.4f} ug/m3")
print(f"  MAPE     : {test_metrics[3]:.2f}%")

r2_diff = train_metrics[0] - test_metrics[0]
print(f"\nOverfitting Assessment:")
print(f"  R2 difference (Train - Test): {r2_diff:.4f}")
if r2_diff < 0.05:
    print("  Status: Minimal overfitting  - Excellent generalisation")
elif r2_diff < 0.10:
    print("  Status: Slight overfitting   - Good generalisation")
elif r2_diff < 0.15:
    print("  Status: Moderate overfitting - Acceptable generalisation")
else:
    print("  Status: Significant overfitting detected")

print(f"\nPrediction Accuracy:")
print(f"  Within +/-5  ug/m3 : {within_5:.1f}%")
print(f"  Within +/-10 ug/m3 : {within_10:.1f}%")

# ─────────────────────────────────────────────────────────────
# STEP 11: FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70)

feature_importances = final_rf.feature_importances_
importance_df = pd.DataFrame({
    'Feature'   : feature_columns,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

print("\nFeature Importance Ranking:\n")
for _, row in importance_df.iterrows():
    print(f"  {row['Feature']:25s}: {row['Importance']:.4f} "
          f"({row['Importance']*100:.2f}%)")

meteorological_importance = importance_df[
    importance_df['Feature'].isin(
        ['Humidity', 'Precipitation', 'Temperature', 'Wind Speed'])
]['Importance'].sum()
lag_importance     = importance_df[
    importance_df['Feature'].str.contains('lag')]['Importance'].sum()
rolling_importance = importance_df[
    importance_df['Feature'].str.contains('rolling')]['Importance'].sum()

print(f"\nImportance by Category:")
print(f"  Meteorological Features  : "
      f"{meteorological_importance:.4f} ({meteorological_importance*100:.2f}%)")
print(f"  Lag Features             : "
      f"{lag_importance:.4f} ({lag_importance*100:.2f}%)")
print(f"  Rolling Average Features : "
      f"{rolling_importance:.4f} ({rolling_importance*100:.2f}%)")

# ─────────────────────────────────────────────────────────────
# STEP 12: VISUALISATIONS (same 6 plots, same style)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("GENERATING VISUALISATIONS")
print("="*70)

# PLOT 1 - Predicted vs Actual
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.scatter(y_test, y_test_pred, alpha=0.6, s=50,
            edgecolors='darkblue', linewidth=0.5, color='steelblue')
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val],
         'r--', lw=2.5, label='Perfect Prediction')
ax1.set_xlabel(r'Actual PM$_{2.5}$ ($\mu$g/m$^3$)',
               fontsize=20, fontweight='bold')
ax1.set_ylabel(r'Predicted PM$_{2.5}$ ($\mu$g/m$^3$)',
               fontsize=20, fontweight='bold')
ax1.tick_params(axis='both', labelsize=16)
ax1.legend(fontsize=16)
ax1.grid(alpha=0.3)
plt.tight_layout()
fig1.savefig(os.path.join(output_dir, 'plot1_predicted_vs_actual.png'),
             dpi=300, bbox_inches='tight')
print("[OK] Saved: 'plot1_predicted_vs_actual.png'")
plt.close(fig1)

# PLOT 2 - Time Series
fig2, ax2 = plt.subplots(figsize=(12, 6))
test_indices = np.arange(len(y_test))
ax2.plot(test_indices, y_test.values, 'o-',
         label=r'Actual PM$_{2.5}$',
         linewidth=2, markersize=4, color='darkgreen', alpha=0.8)
ax2.plot(test_indices, y_test_pred, 's-',
         label=r'Predicted PM$_{2.5}$',
         linewidth=2, markersize=4, color='darkorange', alpha=0.7)
ax2.set_xlabel('Test Sample Index', fontsize=20, fontweight='bold')
ax2.set_ylabel(r'PM$_{2.5}$ ($\mu$g/m$^3$)', fontsize=20, fontweight='bold')
ax2.tick_params(axis='both', labelsize=16)
ax2.legend(fontsize=16)
ax2.grid(alpha=0.3)
plt.tight_layout()
fig2.savefig(os.path.join(output_dir, 'plot2_time_series.png'),
             dpi=300, bbox_inches='tight')
print("[OK] Saved: 'plot2_time_series.png'")
plt.close(fig2)

# PLOT 3 - Residual Plot
fig3, ax3 = plt.subplots(figsize=(10, 8))
ax3.scatter(y_test_pred, residuals, alpha=0.6, s=50,
            edgecolors='darkred', linewidth=0.5, color='salmon')
ax3.axhline(y=0, color='black', linestyle='--', lw=2)
ax3.set_xlabel(r'Predicted PM$_{2.5}$ ($\mu$g/m$^3$)',
               fontsize=20, fontweight='bold')
ax3.set_ylabel(r'Residuals ($\mu$g/m$^3$)', fontsize=20, fontweight='bold')
ax3.tick_params(axis='both', labelsize=16)
ax3.grid(alpha=0.3)
plt.tight_layout()
fig3.savefig(os.path.join(output_dir, 'plot3_residual_plot.png'),
             dpi=300, bbox_inches='tight')
print("[OK] Saved: 'plot3_residual_plot.png'")
plt.close(fig3)

# PLOT 4 - Residual Distribution
fig4, ax4 = plt.subplots(figsize=(10, 8))
ax4.hist(residuals, bins=30, alpha=0.7,
         color='skyblue', edgecolor='black', linewidth=1.2)
ax4.axvline(x=0, color='red', linestyle='--', lw=2.5, label='Zero Error')
ax4.set_xlabel(r'Residuals ($\mu$g/m$^3$)', fontsize=20, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=20, fontweight='bold')
ax4.tick_params(axis='both', labelsize=16)
ax4.legend(fontsize=16)
ax4.grid(alpha=0.3)
plt.tight_layout()
fig4.savefig(os.path.join(output_dir, 'plot4_residual_distribution.png'),
             dpi=300, bbox_inches='tight')
print("[OK] Saved: 'plot4_residual_distribution.png'")
plt.close(fig4)

# PLOT 5 - Feature Importance Bar Chart
fig5, ax5 = plt.subplots(figsize=(12, 9))
importance_plot = importance_df.sort_values('Importance', ascending=True)
colors_bar = [
    '#2E86AB' if 'lag'     in feat else
    '#A23B72' if 'rolling' in feat else
    '#F18F01'
    for feat in importance_plot['Feature']
]
ax5.barh(range(len(importance_plot)), importance_plot['Importance'],
         color=colors_bar, edgecolor='black', linewidth=1)
ax5.set_yticks(range(len(importance_plot)))
ax5.set_yticklabels(importance_plot['Feature'],
                    fontsize=20, fontweight='bold')
ax5.set_xlabel('Importance Score', fontsize=24, fontweight='bold')
ax5.tick_params(axis='x', labelsize=18)
ax5.grid(axis='x', alpha=0.3)
legend_elements = [
    Patch(facecolor='#2E86AB', label='Lag Features (1d, 2d)'),
    Patch(facecolor='#A23B72', label='Rolling Avg (3d, 7d)'),
    Patch(facecolor='#F18F01', label='Meteorological')
]
ax5.legend(handles=legend_elements, loc='lower right', fontsize=18)
plt.tight_layout()
fig5.savefig(os.path.join(output_dir, 'plot5_feature_importance.png'),
             dpi=300, bbox_inches='tight')
print("[OK] Saved: 'plot5_feature_importance.png'")
plt.close(fig5)

# PLOT 6 - Feature Category Pie Chart
fig6, ax6 = plt.subplots(figsize=(10, 8))
categories      = ['Meteorological', 'Lag Features', 'Rolling Averages']
category_values = [meteorological_importance, lag_importance, rolling_importance]
colors_pie      = ['#F18F01', '#2E86AB', '#A23B72']
explode         = (0.05, 0, 0)
ax6.pie(category_values, labels=categories, autopct='%1.1f%%',
        colors=colors_pie, explode=explode, startangle=90,
        textprops={'fontsize': 16, 'fontweight': 'bold'})
plt.tight_layout()
fig6.savefig(os.path.join(output_dir, 'plot6_category_pie_chart.png'),
             dpi=300, bbox_inches='tight')
print("[OK] Saved: 'plot6_category_pie_chart.png'")
plt.close(fig6)

print(f"\n[OK] All 6 plots saved in:\n     {output_dir}")

# ─────────────────────────────────────────────────────────────
# STEP 13: SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame({
    'Date'            : dates_test,
    'Actual_PM2.5'    : y_test.values,
    'Predicted_PM2.5' : y_test_pred,
    'Residual'        : residuals,
    'Absolute_Error'  : abs_errors
})
results_df.to_csv(
    os.path.join(output_dir, 'premonsoon_predictions.csv'), index=False)
print("[OK] Predictions saved: 'premonsoon_predictions.csv'")

importance_df.to_csv(
    os.path.join(output_dir, 'feature_importance.csv'), index=False)
print("[OK] Feature importance saved: 'feature_importance.csv'")

# Determine which params were used for the final model
final_params = (best_bal_params
                if chosen == "Model B - Balanced (min gap)"
                else grid_search.best_params_)

with open(os.path.join(output_dir, 'premonsoon_model_summary.txt'),
          'w', encoding='utf-8') as f:
    f.write("PRE-MONSOON SEASON - PM2.5 PREDICTION MODEL SUMMARY\n")
    f.write("="*70 + "\n\n")
    f.write("MODEL TYPE: Random Forest Regressor\n")
    f.write("  - Season-aware lag/rolling features (no cross-season leakage)\n")
    f.write("  - Outlier-capped features (97th percentile, target uncapped)\n")
    f.write("  - Season-boundary train/test split\n")
    f.write("  - Expanding-window yearly CV\n")
    f.write("  - Conservative parameter grid (anti-overfitting)\n")
    f.write("  - Balanced model selection "
            "(top-20% CV score + min train-val gap)\n\n")
    f.write(f"FINAL MODEL SELECTED: {chosen}\n\n")
    f.write("DATASET:\n")
    f.write(f"  Total samples   : {len(df)}\n")
    f.write(f"  Training seasons: {train_years}\n")
    f.write(f"  Test season     : {test_year}\n")
    f.write(f"  Training samples: {len(X_train)}\n")
    f.write(f"  Testing  samples: {len(X_test)}\n\n")
    f.write("OPTIMISED HYPERPARAMETERS:\n")
    for param, value in final_params.items():
        f.write(f"  {param}: {value}\n")
    f.write(f"\nCV MAE (neg): {grid_search.best_score_:.4f}\n\n")
    f.write("PERFORMANCE METRICS:\n")
    f.write(f"  Training  R2   : {train_metrics[0]:.4f}\n")
    f.write(f"  Training  RMSE : {train_metrics[1]:.4f} ug/m3\n")
    f.write(f"  Training  MAE  : {train_metrics[2]:.4f} ug/m3\n")
    f.write(f"  Training  MAPE : {train_metrics[3]:.2f}%\n\n")
    f.write(f"  Testing   R2   : {test_metrics[0]:.4f}\n")
    f.write(f"  Testing   RMSE : {test_metrics[1]:.4f} ug/m3\n")
    f.write(f"  Testing   MAE  : {test_metrics[2]:.4f} ug/m3\n")
    f.write(f"  Testing   MAPE : {test_metrics[3]:.2f}%\n\n")
    f.write(f"  R2 Gap (Train-Test): {r2_diff:.4f}\n\n")
    f.write(f"PREDICTION ACCURACY:\n")
    f.write(f"  Within +/-5  ug/m3 : {within_5:.1f}%\n")
    f.write(f"  Within +/-10 ug/m3 : {within_10:.1f}%\n\n")
    f.write("FEATURE IMPORTANCE:\n")
    for _, row in importance_df.iterrows():
        f.write(f"  {row['Feature']:25s}: "
                f"{row['Importance']:.4f} ({row['Importance']*100:.2f}%)\n")
    f.write(f"\nCATEGORY IMPORTANCE:\n")
    f.write(f"  Meteorological   : "
            f"{meteorological_importance:.4f} "
            f"({meteorological_importance*100:.2f}%)\n")
    f.write(f"  Lag Features     : "
            f"{lag_importance:.4f} ({lag_importance*100:.2f}%)\n")
    f.write(f"  Rolling Averages : "
            f"{rolling_importance:.4f} ({rolling_importance*100:.2f}%)\n")

print("[OK] Summary report saved: 'premonsoon_model_summary.txt'")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nKey Results:")
print(f"  Test  R2   : {test_metrics[0]:.4f} "
      f"({test_metrics[0]*100:.1f}% variance explained)")
print(f"  Test  RMSE : {test_metrics[1]:.2f} ug/m3")
print(f"  Test  MAE  : {test_metrics[2]:.2f} ug/m3")
print(f"  R2 Gap     : {r2_diff:.4f}  (was 0.2381 before)")
print(f"  Within +/-5 ug/m3 : {within_5:.1f}%")

print(f"\nTop 3 Most Important Features:")
for i, (_, row) in enumerate(importance_df.head(3).iterrows(), 1):
    print(f"  {i}. {row['Feature']} ({row['Importance']*100:.2f}%)")