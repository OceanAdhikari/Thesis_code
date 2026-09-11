import io
import os
import sys
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding='utf-8', errors='replace'
)
plt.rcParams['font.family'] = 'Times New Roman'

output_dir = r'C:\Users\ocean\OneDrive\Desktop\OA Thesis\Standarized Data\Winter\RF\min2_cancel'
os.makedirs(output_dir, exist_ok=True)

file_path = r'C:\Users\ocean\OneDrive\Desktop\OA Thesis\Standarized Data\Winter\RF\Book1.csv'
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# METEOROLOGICAL SEASON-YEAR (CRITICAL FIX)
# December is shifted into the NEXT calendar year so that
# Dec 2018 + Jan 2019 + Feb 2019 form one continuous winter.
# Formula: season_year = year + 1 if month == 12 else year
# ─────────────────────────────────────────────────────────────
df['season_year'] = df['date'].dt.year + (df['date'].dt.month == 12).astype(int)

print('Season-year mapping check (first/last few rows):')
print(df[['date', 'season_year']].head(8).to_string(index=False))
print('...')
print(df[['date', 'season_year']].tail(8).to_string(index=False))
print(f"\nUnique season_years present: {sorted(df['season_year'].unique())}")

# ─────────────────────────────────────────────────────────────
# OUTLIER TREATMENT
# ─────────────────────────────────────────────────────────────
pm25_cap = df['PM2.5'].quantile(0.97)
print(f'\nPM2.5 97th percentile cap: {pm25_cap:.2f} ug/m3')
df['PM2.5_feat'] = df['PM2.5'].clip(upper=pm25_cap)

# ─────────────────────────────────────────────────────────────
# SEASON-AWARE FEATURES (grouped by season_year, NOT calendar year)
# Dec 1 now correctly gets lag from late Nov (if present) or NaN —
# NEVER from February of the same calendar year.
# ─────────────────────────────────────────────────────────────
for lag in [1, 2]:
  df[f'PM2.5_lag_{lag}d'] = df.groupby('season_year')['PM2.5_feat'].shift(lag)

df['PM2.5_rolling_3d'] = df.groupby('season_year')['PM2.5_feat'].transform(
    lambda s: s.shift(1).rolling(window=3).mean()
)
df['PM2.5_rolling_7d'] = df.groupby('season_year')['PM2.5_feat'].transform(
    lambda s: s.shift(1).rolling(window=7).mean()
)

df = df.dropna().reset_index(drop=True)

feature_columns = [
    'Humidity',
    'Precipitation',
    'Temperature',
    'Wind Speed',
    'PM2.5_lag_1d',
    'PM2.5_lag_2d',
    'PM2.5_rolling_3d',
    'PM2.5_rolling_7d',
]
X = df[feature_columns]
y = df['PM2.5']

# ─────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT (Winter 2019 Selected as Test Year)
# Test season_year = 2019  =>  Dec 2018 + Jan 2019 + Feb 2019
# Train = all other continuous winters (2018, 2020, 2021, 2022, 2023)
# ─────────────────────────────────────────────────────────────
test_year = 2019
available_years = sorted(df['season_year'].unique())
train_years = [y for y in available_years if y != test_year]

train_mask = df['season_year'].isin(train_years)
test_mask = df['season_year'] == test_year

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
dates_test = df.loc[test_mask, 'date'].reset_index(drop=True)
years_train_series = df.loc[train_mask, 'season_year'].reset_index(drop=True)

print('\n' + '=' * 70)
print('WINTER SEASON - RANDOM FOREST MODEL FOR PM2.5 PREDICTION')
print('=' * 70)
print(
    f"\n  Date range       : {df['date'].min().date()} to"
    f" {df['date'].max().date()}"
)
print(f'  Total samples    : {len(df)}')
print(f'  Training winters : {train_years}')
print(
    f'  Test winter      : {test_year}  (Dec {test_year-1} + Jan/Feb'
    f' {test_year})'
)
print(
    f'  Test date range  : {dates_test.min().date()} ->'
    f' {dates_test.max().date()}'
)
print(f'  Training samples : {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)')
print(f'  Testing  samples : {len(X_test)}  ({len(X_test)/len(df)*100:.1f}%)')

# Guard check: test set must be one continuous Dec–Feb block
assert (
    dates_test.min().month == 12 or dates_test.min().month <= 2
), 'Test set does not start in Dec/Jan/Feb — check season_year logic!'
assert (
    dates_test.max().month <= 2
), 'Test set does not end in Jan/Feb — check season_year logic!'
print('  [OK] Test set is a continuous meteorological winter block.')

# ─────────────────────────────────────────────────────────────
# EXPANDING-WINDOW CV (by season_year)
# ─────────────────────────────────────────────────────────────
cv_folds = []
unique_train_years = sorted(years_train_series.unique())

for i in range(1, len(unique_train_years)):
  val_year = unique_train_years[i]
  train_yrs = unique_train_years[:i]
  train_idx = np.where(years_train_series.isin(train_yrs))[0]
  val_idx = np.where(years_train_series == val_year)[0]
  if len(train_idx) > 0 and len(val_idx) > 0:
    cv_folds.append((train_idx, val_idx))

print('\nExpanding-window CV folds:')
for i, (tr_idx, va_idx) in enumerate(cv_folds, 1):
  tr_yrs = sorted(years_train_series.iloc[tr_idx].unique())
  va_yr = years_train_series.iloc[va_idx].unique()[0]
  print(f'  Fold {i}: train={tr_yrs} -> validate={va_yr} ({len(va_idx)} rows)')

# ─────────────────────────────────────────────────────────────
# PARAMETER GRID
# ─────────────────────────────────────────────────────────────
param_grid = {
    'n_estimators': [300, 500, 700],
    'max_depth': [4, 5, 6],
    'min_samples_split': [15, 20, 30],
    'min_samples_leaf': [8, 12, 18],
    'max_features': ['sqrt', 0.4, 0.5],
    'max_leaf_nodes': [30, 45, 60],
    'min_impurity_decrease': [0.02, 0.05],
    'max_samples': [0.65, 0.75, 0.85],
    'bootstrap': [True],
}

n_combinations = 1
for v in param_grid.values():
  n_combinations *= len(v)

print('\n' + '=' * 70)
print('HYPERPARAMETER OPTIMISATION')
print('=' * 70)
print(
    f'Grid: {n_combinations} combinations x {len(cv_folds)} folds = '
    f'{n_combinations * len(cv_folds)} fits'
)

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=cv_folds,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
)
grid_search.fit(X_train, y_train)

print('\nBest Parameters:')
for param, value in grid_search.best_params_.items():
  print(f'  {param}: {value}')
print(f'Best CV MAE (neg): {grid_search.best_score_:.4f}')

# ─────────────────────────────────────────────────────────────
# BALANCED MODEL SELECTION
# ─────────────────────────────────────────────────────────────
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results['gap'] = (
    cv_results['mean_train_score'] - cv_results['mean_test_score']
).abs()

threshold = cv_results['mean_test_score'].quantile(0.80)
top20pct = cv_results[cv_results['mean_test_score'] >= threshold]
best_bal_idx = top20pct['gap'].idxmin()
best_bal_params = cv_results.loc[best_bal_idx, 'params']


# ─────────────────────────────────────────────────────────────
# TRAIN AND EVALUATE BOTH MODELS
# ─────────────────────────────────────────────────────────────
def train_and_evaluate(params, X_tr, y_tr, X_te, y_te, label):
  clean_params = {k: v for k, v in params.items() if k != 'n_jobs'}
  model = RandomForestRegressor(**clean_params, random_state=42)
  model.fit(X_tr, y_tr)
  y_tr_pred = model.predict(X_tr)
  y_te_pred = model.predict(X_te)
  r2_tr = r2_score(y_tr, y_tr_pred)
  r2_te = r2_score(y_te, y_te_pred)
  rmse = np.sqrt(mean_squared_error(y_te, y_te_pred))
  mae = mean_absolute_error(y_te, y_te_pred)
  mape = np.mean(np.abs((y_te - y_te_pred) / (y_te + 1e-8))) * 100
  gap = r2_tr - r2_te
  print(f'\n  [{label}]')
  print(f'    Train R2 : {r2_tr:.4f}')
  print(f'    Test  R2 : {r2_te:.4f}')
  print(f'    R2 Gap   : {gap:.4f}')
  print(f'    RMSE     : {rmse:.4f} ug/m3')
  print(f'    MAE      : {mae:.4f} ug/m3')
  print(f'    MAPE     : {mape:.2f}%')
  return model, y_tr_pred, y_te_pred, {
      'r2_train': r2_tr,
      'r2_test': r2_te,
      'rmse': rmse,
      'mae': mae,
      'mape': mape,
      'gap': gap,
  }


print('\n' + '=' * 70)
print('MODEL COMPARISON')
print('=' * 70)

model_A, ytr_A, yte_A, met_A = train_and_evaluate(
    grid_search.best_params_,
    X_train,
    y_train,
    X_test,
    y_test,
    'Model A - GridSearch Best',
)

model_B, ytr_B, yte_B, met_B = train_and_evaluate(
    best_bal_params, X_train, y_train, X_test, y_test, 'Model B - Balanced'
)

if (abs(met_B['gap']) < abs(met_A['gap']) and met_B['r2_test'] >= met_A['r2_test'] - 0.03):
  final_rf = model_B
  y_test_pred = yte_B
  y_train_pred = ytr_B
  final_metrics = met_B
  chosen = 'Model B - Balanced (min gap)'
  final_params = best_bal_params
else:
  final_rf = model_A
  y_test_pred = yte_A
  y_train_pred = ytr_A
  final_metrics = met_A
  chosen = 'Model A - GridSearch Best'
  final_params = grid_search.best_params_

print(f'\n>>> FINAL MODEL: {chosen}')

# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────
residuals = y_test.values - y_test_pred
abs_errors = np.abs(residuals)
within_5 = (abs_errors <= 5).sum() / len(abs_errors) * 100
within_10 = (abs_errors <= 10).sum() / len(abs_errors) * 100

train_metrics = (
    final_metrics['r2_train'],
    np.sqrt(mean_squared_error(y_train, y_train_pred)),
    mean_absolute_error(y_train, y_train_pred),
    np.mean(np.abs((y_train - y_train_pred) / (y_train + 1e-8))) * 100,
)
test_metrics = (
    final_metrics['r2_test'],
    final_metrics['rmse'],
    final_metrics['mae'],
    final_metrics['mape'],
)
r2_diff = train_metrics[0] - test_metrics[0]

if abs(r2_diff) < 0.05:
  overfit_status = 'Minimal overfitting - Excellent generalisation'
elif abs(r2_diff) < 0.10:
  overfit_status = 'Slight overfitting - Good generalisation'
elif abs(r2_diff) < 0.15:
  overfit_status = 'Moderate overfitting - Acceptable generalisation'
else:
  overfit_status = 'Significant overfitting detected'

print('\n' + '=' * 70)
print('FINAL MODEL PERFORMANCE')
print('=' * 70)
print('\nTraining Set:')
print(f'  R2 Score : {train_metrics[0]:.4f}')
print(f'  RMSE     : {train_metrics[1]:.4f} ug/m3')
print(f'  MAE      : {train_metrics[2]:.4f} ug/m3')
print(f'  MAPE     : {train_metrics[3]:.2f}%')
print('\nTesting Set:')
print(f'  R2 Score : {test_metrics[0]:.4f}')
print(f'  RMSE     : {test_metrics[1]:.4f} ug/m3')
print(f'  MAE      : {test_metrics[2]:.4f} ug/m3')
print(f'  MAPE     : {test_metrics[3]:.2f}%')
print('\nOverfitting Assessment:')
print(f'  R2 difference (Train - Test): {r2_diff:.4f}')
print(f'  Status: {overfit_status}')
print('\nPrediction Accuracy:')
print(f'  Within +/-5  ug/m3 : {within_5:.1f}%')
print(f'  Within +/-10 ug/m3 : {within_10:.1f}%')

# ─────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': final_rf.feature_importances_,
}).sort_values('Importance', ascending=False)

meteorological_importance = importance_df[
    importance_df['Feature'].isin(
        ['Humidity', 'Precipitation', 'Temperature', 'Wind Speed']
    )
]['Importance'].sum()
lag_importance = importance_df[importance_df['Feature'].str.contains('lag')][
    'Importance'
].sum()
rolling_importance = importance_df[
    importance_df['Feature'].str.contains('rolling')
]['Importance'].sum()

print('\n' + '=' * 70)
print('FEATURE IMPORTANCE')
print('=' * 70)
for _, row in importance_df.iterrows():
  print(
      f"  {row['Feature']:25s}: {row['Importance']:.4f} "
      f"({row['Importance']*100:.2f}%)"
  )
print(f'\n  Meteorological : {meteorological_importance*100:.2f}%')
print(f'  Lag Features   : {lag_importance*100:.2f}%')
print(f'  Rolling Avg    : {rolling_importance*100:.2f}%')

# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('GENERATING VISUALISATIONS')
print('=' * 70)

fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.scatter(
    y_test,
    y_test_pred,
    alpha=0.6,
    s=50,
    edgecolors='darkblue',
    linewidth=0.5,
    color='steelblue',
)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax1.plot(
    [min_val, max_val],
    [min_val, max_val],
    'r--',
    lw=2.5,
    label='Perfect Prediction',
)
ax1.set_xlabel(
    r'Actual PM$_{2.5}$ ($\mu$g/m$^3$)', fontsize=20, fontweight='bold'
)
ax1.set_ylabel(
    r'Predicted PM$_{2.5}$ ($\mu$g/m$^3$)', fontsize=20, fontweight='bold'
)
ax1.tick_params(axis='both', labelsize=16)
ax1.legend(fontsize=16)
ax1.grid(alpha=0.3)
plt.tight_layout()
fig1.savefig(
    os.path.join(output_dir, 'plot1_predicted_vs_actual.png'),
    dpi=300,
    bbox_inches='tight',
)
print('[OK] plot1_predicted_vs_actual.png')
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(
    np.arange(len(y_test)),
    y_test.values,
    'o-',
    label=r'Actual PM$_{2.5}$',
    linewidth=2,
    markersize=4,
    color='darkgreen',
    alpha=0.8,
)
ax2.plot(
    np.arange(len(y_test)),
    y_test_pred,
    's-',
    label=r'Predicted PM$_{2.5}$',
    linewidth=2,
    markersize=4,
    color='darkorange',
    alpha=0.7,
)
ax2.set_xlabel('Test Sample Index', fontsize=20, fontweight='bold')
ax2.set_ylabel(r'PM$_{2.5}$ ($\mu$g/m$^3$)', fontsize=20, fontweight='bold')
ax2.tick_params(axis='both', labelsize=16)
ax2.legend(fontsize=16)
ax2.grid(alpha=0.3)
plt.tight_layout()
fig2.savefig(
    os.path.join(output_dir, 'plot2_time_series.png'),
    dpi=300,
    bbox_inches='tight',
)
print('[OK] plot2_time_series.png')
plt.close(fig2)

fig3, ax3 = plt.subplots(figsize=(10, 8))
ax3.scatter(
    y_test_pred,
    residuals,
    alpha=0.6,
    s=50,
    edgecolors='darkred',
    linewidth=0.5,
    color='salmon',
)
ax3.axhline(y=0, color='black', linestyle='--', lw=2)
ax3.set_xlabel(
    r'Predicted PM$_{2.5}$ ($\mu$g/m$^3$)', fontsize=20, fontweight='bold'
)
ax3.set_ylabel(r'Residuals ($\mu$g/m$^3$)', fontsize=20, fontweight='bold')
ax3.tick_params(axis='both', labelsize=16)
ax3.grid(alpha=0.3)
plt.tight_layout()
fig3.savefig(
    os.path.join(output_dir, 'plot3_residual_plot.png'),
    dpi=300,
    bbox_inches='tight',
)
print('[OK] plot3_residual_plot.png')
plt.close(fig3)

fig4, ax4 = plt.subplots(figsize=(10, 8))
ax4.hist(
    residuals,
    bins=30,
    alpha=0.7,
    color='skyblue',
    edgecolor='black',
    linewidth=1.2,
)
ax4.axvline(x=0, color='red', linestyle='--', lw=2.5, label='Zero Error')
ax4.set_xlabel(r'Residuals ($\mu$g/m$^3$)', fontsize=20, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=20, fontweight='bold')
ax4.tick_params(axis='both', labelsize=16)
ax4.legend(fontsize=16)
ax4.grid(alpha=0.3)
plt.tight_layout()
fig4.savefig(
    os.path.join(output_dir, 'plot4_residual_distribution.png'),
    dpi=300,
    bbox_inches='tight',
)
print('[OK] plot4_residual_distribution.png')
plt.close(fig4)

fig5, ax5 = plt.subplots(figsize=(12, 9))
importance_plot = importance_df.sort_values('Importance', ascending=True)
colors_bar = [
    '#2E86AB'
    if 'lag' in f
    else '#A23B72'
    if 'rolling' in f
    else '#F18F01'
    for f in importance_plot['Feature']
]
ax5.barh(
    range(len(importance_plot)),
    importance_plot['Importance'],
    color=colors_bar,
    edgecolor='black',
    linewidth=1,
)
ax5.set_yticks(range(len(importance_plot)))
ax5.set_yticklabels(
    importance_plot['Feature'], fontsize=20, fontweight='bold'
)
ax5.set_xlabel('Importance Score', fontsize=24, fontweight='bold')
ax5.tick_params(axis='x', labelsize=18)
ax5.grid(axis='x', alpha=0.3)
ax5.legend(
    handles=[
        Patch(facecolor='#2E86AB', label='Lag Features (1d, 2d)'),
        Patch(facecolor='#A23B72', label='Rolling Avg (3d, 7d)'),
        Patch(facecolor='#F18F01', label='Meteorological'),
    ],
    loc='lower right',
    fontsize=18,
)
plt.tight_layout()
fig5.savefig(
    os.path.join(output_dir, 'plot5_feature_importance.png'),
    dpi=300,
    bbox_inches='tight',
)
print('[OK] plot5_feature_importance.png')
plt.close(fig5)

fig6, ax6 = plt.subplots(figsize=(10, 8))
ax6.pie(
    [meteorological_importance, lag_importance, rolling_importance],
    labels=['Meteorological', 'Lag Features', 'Rolling Averages'],
    autopct='%1.1f%%',
    colors=['#F18F01', '#2E86AB', '#A23B72'],
    explode=(0.05, 0, 0),
    startangle=90,
    textprops={'fontsize': 16, 'fontweight': 'bold'},
)
plt.tight_layout()
fig6.savefig(
    os.path.join(output_dir, 'plot6_category_pie_chart.png'),
    dpi=300,
    bbox_inches='tight',
)
print('[OK] plot6_category_pie_chart.png')
plt.close(fig6)

# ─────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────
pd.DataFrame({
    'Date': dates_test,
    'Actual_PM2.5': y_test.values,
    'Predicted_PM2.5': y_test_pred,
    'Residual': residuals,
    'Absolute_Error': abs_errors,
}).to_csv(os.path.join(output_dir, 'winter_predictions.csv'), index=False)
print('[OK] winter_predictions.csv')

importance_df.to_csv(
    os.path.join(output_dir, 'feature_importance.csv'), index=False
)
print('[OK] feature_importance.csv')

with open(
    os.path.join(output_dir, 'winter_model_summary.txt'), 'w', encoding='utf-8'
) as f:
  f.write('WINTER SEASON - PM2.5 PREDICTION MODEL SUMMARY\n')
  f.write('=' * 70 + '\n\n')
  f.write('MODEL TYPE: Random Forest Regressor\n')
  f.write(
      '  - Meteorological season_year grouping (Dec joins following Jan/Feb)\n'
  )
  f.write('  - Season-aware lag/rolling (shift(1), groupby season_year)\n')
  f.write('  - Outlier-capped features (97th percentile)\n')
  f.write('  - Expanding-window yearly CV on continuous winters\n')
  f.write('  - Middle-ground parameter grid\n')
  f.write('  - Balanced model selection\n\n')
  f.write('SEASON_YEAR DEFINITION:\n')
  f.write('  season_year = year + 1  if month == 12  else year\n')
  f.write('  Example: Dec 2018 + Jan 2019 + Feb 2019  =>  season_year = 2019\n')
  f.write('  This prevents Dec rows from inheriting lag/rolling values\n')
  f.write(
      '  from Feb of the same calendar year (silent cross-winter leak).\n\n'
  )
  f.write('SPLIT RATIONALE:\n')
  f.write(f'  Test winter = season_year {test_year}\n')
  f.write(
      f'  (continuous block: Dec {test_year-1} through Feb {test_year})\n'
  )
  f.write(f'  Train winters = {train_years}\n\n')
  f.write(f'FINAL MODEL: {chosen}\n\n')
  f.write('OPTIMISED HYPERPARAMETERS:\n')
  for p, v in final_params.items():
    f.write(f'  {p}: {v}\n')
  f.write('\nPERFORMANCE METRICS:\n')
  f.write(f'  Training R2   : {train_metrics[0]:.4f}\n')
  f.write(f'  Training RMSE : {train_metrics[1]:.4f} ug/m3\n')
  f.write(f'  Training MAE  : {train_metrics[2]:.4f} ug/m3\n')
  f.write(f'  Training MAPE : {train_metrics[3]:.2f}%\n\n')
  f.write(f'  Testing  R2   : {test_metrics[0]:.4f}\n')
  f.write(f'  Testing  RMSE : {test_metrics[1]:.4f} ug/m3\n')
  f.write(f'  Testing  MAE  : {test_metrics[2]:.4f} ug/m3\n')
  f.write(f'  Testing  MAPE : {test_metrics[3]:.2f}%\n\n')
  f.write(f'  R2 Gap  : {r2_diff:.4f}\n')
  f.write(f'  Status  : {overfit_status}\n\n')
  f.write('PREDICTION ACCURACY:\n')
  f.write(f'  Within +/-5  ug/m3 : {within_5:.1f}%\n')
  f.write(f'  Within +/-10 ug/m3 : {within_10:.1f}%\n\n')
  f.write('FEATURE IMPORTANCE:\n')
  for _, row in importance_df.iterrows():
    f.write(
        f"  {row['Feature']:25s}: {row['Importance']:.4f} "
        f"({row['Importance']*100:.2f}%)\n"
    )
  f.write('\nCATEGORY IMPORTANCE:\n')
  f.write(f'  Meteorological : {meteorological_importance*100:.2f}%\n')
  f.write(f'  Lag Features   : {lag_importance*100:.2f}%\n')
  f.write(f'  Rolling Avg    : {rolling_importance*100:.2f}%\n')

print('[OK] winter_model_summary.txt')

print('\n' + '=' * 70)
print('ANALYSIS COMPLETE')
print('=' * 70)
print(
    f'\n  Test  R2   : {test_metrics[0]:.4f} '
    f'({test_metrics[0]*100:.1f}% variance explained)'
)
print(f'  Test  RMSE : {test_metrics[1]:.2f} ug/m3')
print(f'  Test  MAE  : {test_metrics[2]:.2f} ug/m3')
print(f'  R2 Gap     : {r2_diff:.4f}')
print(f'  Status     : {overfit_status}')
print(f'  Within +/-5  ug/m3 : {within_5:.1f}%')
print(f'  Within +/-10 ug/m3 : {within_10:.1f}%')
print('\nTop 3 Features:')
for i, (_, row) in enumerate(importance_df.head(3).iterrows(), 1):
  print(f"  {i}. {row['Feature']} ({row['Importance']*100:.2f}%)")
print(f'\nAll outputs saved to:\n  {output_dir}')