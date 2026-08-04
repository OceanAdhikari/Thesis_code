import os
import sys
import io
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
plt.rcParams['font.family'] = 'Times New Roman'

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
base_dir   = r"C:\Users\ocean\OneDrive\Desktop\OA Thesis\High Pollution cases\Precipitation\5mm_lag"
output_dir = base_dir
os.makedirs(output_dir, exist_ok=True)

file_path = os.path.join(base_dir, "cdd_pm25_result.csv")

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print("="*70)
print("HIGH POLLUTION CASES - FEATURE IMPORTANCE ANALYSIS")
print("Purpose: Identify which physical factor drives PM2.5 most")
print("="*70)
print(f"\n  Total samples: {len(df)}")
print(f"  Date range   : {df['Date'].min().date()} to {df['Date'].max().date()}")

# ─────────────────────────────────────────────────────────────
# 5 PHYSICAL FEATURES
# ─────────────────────────────────────────────────────────────
feature_columns = [
    'Humidity',
    'Precipitation',
    'Temperature',
    'Wind Speed',
    'Days_since_last_precipitation'
]

X = df[feature_columns]
y = df['Pm2.5']

# ─────────────────────────────────────────────────────────────
# FEATURE CORRELATIONS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PEARSON CORRELATION WITH PM2.5")
print("="*70)
for col in feature_columns:
    corr = df[col].corr(df['Pm2.5'])
    print(f"  {col:35s}: {corr:+.4f}")

# ─────────────────────────────────────────────────────────────
# TRAIN RANDOM FOREST (on ALL data — for importance only)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TRAINING RANDOM FOREST FOR FEATURE IMPORTANCE")
print("="*70)

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=6,
    min_samples_leaf=8,
    min_samples_split=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)
print("[OK] Model trained.")

# ─────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
importance_df = pd.DataFrame({
    'Feature'   : feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + "="*70)
print("FEATURE IMPORTANCE RANKING")
print("="*70)
for i, row in enumerate(importance_df.itertuples(index=False), 1):
    star = "  <-- CDD" if row.Feature == 'Days_since_last_precipitation' else ""
    print(f"  {i}. {row.Feature:35s}: {row.Importance*100:6.2f}%{star}")

# ─────────────────────────────────────────────────────────────
# PLOT 1: BAR CHART
# ─────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(12, 8))
importance_plot = importance_df.sort_values('Importance', ascending=True)

colors = ['#2ECC71' if f == 'Days_since_last_precipitation' else '#F18F01'
          for f in importance_plot['Feature']]

bars = ax1.barh(range(len(importance_plot)), importance_plot['Importance'],
                color=colors, edgecolor='black', linewidth=1.2)

ax1.set_yticks(range(len(importance_plot)))
ax1.set_yticklabels(importance_plot['Feature'],
                    fontsize=18, fontweight='bold')
ax1.set_xlabel('Importance Score', fontsize=22, fontweight='bold')
ax1.tick_params(axis='x', labelsize=16)
ax1.grid(axis='x', alpha=0.3)
ax1.legend(handles=[
    Patch(facecolor='#F18F01', label='Meteorological'),
    Patch(facecolor='#2ECC71', label='CDD (Days Since Last Precipitation)')
], loc='lower right', fontsize=16)
plt.tight_layout()
fig1.savefig(os.path.join(output_dir, 'feature_importance_bar.png'),
             dpi=300, bbox_inches='tight')
print("[OK] feature_importance_bar.png")
plt.close(fig1)

# ─────────────────────────────────────────────────────────────
# PLOT 2: PIE CHART
# ─────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 8))
colors_pie = ['#2ECC71' if f == 'Days_since_last_precipitation' else '#F18F01'
              for f in importance_df['Feature']]

ax2.pie(
    importance_df['Importance'],
    labels=importance_df['Feature'],
    autopct='%1.1f%%',
    colors=colors_pie,
    startangle=90,
    textprops={'fontsize': 13, 'fontweight': 'bold'}
)
plt.tight_layout()
fig2.savefig(os.path.join(output_dir, 'feature_importance_pie.png'),
             dpi=300, bbox_inches='tight')
print("[OK] feature_importance_pie.png")
plt.close(fig2)

# ─────────────────────────────────────────────────────────────
# SAVE CSV + SUMMARY
# ─────────────────────────────────────────────────────────────
importance_df.to_csv(
    os.path.join(output_dir, 'feature_importance.csv'), index=False)
print("[OK] feature_importance.csv")

with open(os.path.join(output_dir, 'feature_importance_summary.txt'),
          'w', encoding='utf-8') as f:
    f.write("HIGH POLLUTION CASES - FEATURE IMPORTANCE ANALYSIS\n")
    f.write("="*70 + "\n\n")
    f.write("PURPOSE: Identify most important physical factor for PM2.5\n")
    f.write("METHOD : Random Forest on 5 physical features (all data)\n")
    f.write("         NO lag features, NO rolling averages\n\n")
    f.write(f"Total samples: {len(df)}\n")
    f.write(f"Date range   : {df['Date'].min().date()} to {df['Date'].max().date()}\n\n")
    f.write("FEATURES ANALYSED (5):\n")
    for i, feat in enumerate(feature_columns, 1):
        f.write(f"  {i}. {feat}\n")
    f.write("\nPEARSON CORRELATIONS:\n")
    for col in feature_columns:
        corr = df[col].corr(df['Pm2.5'])
        f.write(f"  {col:35s}: {corr:+.4f}\n")
    f.write("\nFEATURE IMPORTANCE RANKING:\n")
    for i, row in enumerate(importance_df.itertuples(index=False), 1):
        star = "  <-- CDD" if row.Feature == 'Days_since_last_precipitation' else ""
        f.write(f"  {i}. {row.Feature:35s}: {row.Importance*100:6.2f}%{star}\n")

print("[OK] feature_importance_summary.txt")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\n  Most important factor : {importance_df.iloc[0]['Feature']} "
      f"({importance_df.iloc[0]['Importance']*100:.2f}%)")
print(f"  Least important factor: {importance_df.iloc[-1]['Feature']} "
      f"({importance_df.iloc[-1]['Importance']*100:.2f}%)")
print(f"\n  CDD ranked            : "
      f"{importance_df.reset_index(drop=True).index[importance_df['Feature']=='Days_since_last_precipitation'].tolist()[0]+1} "
      f"out of 5")
print(f"  CDD importance        : "
      f"{importance_df[importance_df['Feature']=='Days_since_last_precipitation']['Importance'].values[0]*100:.2f}%")