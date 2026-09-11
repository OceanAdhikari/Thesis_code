import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os

plt.rcParams['font.family'] = 'Times New Roman'
# Make math text (e.g. PM$_{2.5}$, $\mu$g/m$^3$) render in a Times-like serif
# font too, so it matches the rest of the labels instead of falling back to
# the default STIX/DejaVu math font.
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# ================== CONFIGURATION ==================
file_path = r'C:\Users\ocean\OneDrive\Desktop\OA Thesis\High Pollution cases\cdd_pm25_result.csv'

output_folder   = r'C:\Users\ocean\OneDrive\Desktop\OA Thesis\Python5'
output_filename = 'pm25_cdd_clean_plot.png'

date_col = 'Date'
pm_col   = 'Pm2.5'
cdd_col  = 'CDD'
# ===================================================

# ── 1. Load & parse dates ──────────────────────────
df = pd.read_csv(file_path)
df[date_col] = pd.to_datetime(df[date_col], format='%m/%d/%Y', errors='coerce')
df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

# ── 2. Clip to Dec 2017 - May 2023 ────────────────
df = df[
    (df[date_col] >= '2017-12-01') &
    (df[date_col] <= '2023-05-31')
].reset_index(drop=True)

# ── 3. Remove monsoon months (June-September) ──────
EXCLUDED_MONTHS = [6, 7, 8, 9]
df = df[~df[date_col].dt.month.isin(EXCLUDED_MONTHS)].reset_index(drop=True)

# ── 4. Fill missing CDD ────────────────────────────
df[cdd_col] = df[cdd_col].fillna(0)

# ── 5. Remove PM2.5 outliers using IQR ────────────
Q1  = df[pm_col].quantile(0.25)
Q3  = df[pm_col].quantile(0.75)
IQR = Q3 - Q1
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

outlier_mask = (df[pm_col] < lower_fence) | (df[pm_col] > upper_fence)
n_removed    = outlier_mask.sum()
print(f"IQR bounds  ->  lower: {lower_fence:.2f}  |  upper: {upper_fence:.2f}")
print(f"Outliers removed: {n_removed} rows")

df = df[~outlier_mask].copy().reset_index(drop=True)
df['CDD_inverted'] = -df[cdd_col]

# ── 6. Integer x-axis (no gaps) ───────────────────
df['x'] = np.arange(len(df))

df['ym'] = df[date_col].dt.to_period('M')
month_first = df.groupby('ym', sort=False)['x'].first().sort_index()

# ── Tick labels: numeric month numbers only ────────
LABEL_MONTHS = {10, 11, 12, 1, 2, 3, 4, 5}
tick_pos    = []
tick_labels = []

for period, xpos in list(month_first.items()):
    tick_pos.append(xpos)
    if period.month in LABEL_MONTHS:
        tick_labels.append(str(period.month))  # ← numeric: 12, 1, 2 ... 10, 11
    else:
        tick_labels.append('')

# ── 7. Figure ──────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(16, 7))

color_pm  = '#1f77b4'
color_cdd = '#d62728'

# Thinner lines than the previous pass — the fill/lines were reading too
# heavy at print size.
LW_PM  = 1.6   # was 2.2
LW_CDD = 1.8   # was 2.6

# Tick label sizes: PM (left) axis kept as-is (this is the visual "base").
# X-axis and CDD (right) axis are grouped together and set smaller/equal to
# each other, distinct from the PM axis.
TICK_SIZE_PM_Y = 20   # unchanged
TICK_SIZE_X    = 20   # was 20
TICK_SIZE_CDD  = 20   # was 20

# Alternating year shading
years = sorted(df[date_col].dt.year.unique())
for i, yr in enumerate(years):
    yr_rows = df[df[date_col].dt.year == yr]
    if yr_rows.empty:
        continue
    x0, x1 = yr_rows['x'].iloc[0], yr_rows['x'].iloc[-1]
    if i % 2 == 0:
        ax1.axvspan(x0, x1, color='#f0f4f8', alpha=0.8, zorder=0)

# Vertical dashed year separators
for yr in years[1:]:
    yr_rows = df[df[date_col].dt.year == yr]
    if not yr_rows.empty:
        ax1.axvline(yr_rows['x'].iloc[0], color='#aaaaaa',
                    linewidth=0.8, linestyle='--', alpha=0.6, zorder=1)

# ─── LEFT axis: PM2.5 ─────────────────────────────
ax1.plot(df['x'], df[pm_col],
         color=color_pm, linewidth=LW_PM, zorder=3, label='PM$_{2.5}$')
ax1.fill_between(df['x'], df[pm_col],
                 alpha=0.13, color=color_pm, zorder=2)

ax1.set_ylabel(r'PM$_{2.5}$ Concentration ($\mu$g/m$^3$)',
               color=color_pm, fontsize=26, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_pm, labelsize=TICK_SIZE_PM_Y)

pm_max = df[pm_col].max()
ax1.set_ylim(bottom=0, top=pm_max * 1.22)
ax1.grid(True, alpha=0.3, linestyle='--', zorder=1)
ax1.set_axisbelow(True)

# ─── RIGHT axis: CDD ──────────────────────────────
ax2 = ax1.twinx()
ax2.plot(df['x'], df['CDD_inverted'],
         color=color_cdd, linewidth=LW_CDD, zorder=3, label='Consecutive Dry Days (CDD)')
ax2.fill_between(df['x'], df['CDD_inverted'],
                 alpha=0.10, color=color_cdd, zorder=2)

ax2.set_ylabel('Consecutive Dry Days (CDD)',
               color=color_cdd, fontsize=26, fontweight='bold')
ax2.invert_yaxis()

max_cdd = int(df[cdd_col].max()) if len(df) else 30
step    = 10 if max_cdd > 60 else 5
yticks  = np.arange(0, max_cdd + step + 1, step)
ax2.set_yticks(-yticks)
ax2.set_yticklabels(yticks, fontsize=TICK_SIZE_CDD)
ax2.tick_params(axis='y', labelcolor=color_cdd)
ax2.grid(False)
ax2.set_ylim(top=-(max_cdd * 1.35), bottom=5)

# ─── X-axis ticks ─────────────────────────────────
ax1.set_xlim(df['x'].iloc[0] - 3, df['x'].iloc[-1] + 3)
ax1.set_xticks(tick_pos)
ax1.set_xticklabels(tick_labels, fontsize=TICK_SIZE_X, ha='center', va='top')
ax1.tick_params(axis='x', which='major', length=4, pad=6)

ax1.set_xlabel('Month', fontsize=26, fontweight='bold', labelpad=14)

# Force bold on every tick label — x-axis, and both y-axes (PM left, CDD
# right). tick_params/set_yticklabels don't set font weight, so it's
# applied directly on the tick label Text objects.
for label in ax1.get_xticklabels():
    label.set_fontweight('bold')
for label in ax1.get_yticklabels():
    label.set_fontweight('bold')
for label in ax2.get_yticklabels():
    label.set_fontweight('bold')

# ─── Legend ───────────────────────────────────────
line_pm  = mlines.Line2D([], [], color=color_pm,  linewidth=2.5, label='PM$_{2.5}$')
line_cdd = mlines.Line2D([], [], color=color_cdd, linewidth=2.5, label='Consecutive Dry Days (CDD)')

legend = ax1.legend(
    handles=[line_pm, line_cdd],
    loc='upper right',
    bbox_to_anchor=(1.0, 1.0),
    ncol=1,
    prop={'family': 'Times New Roman', 'size': 20},
    framealpha=1.0,
    edgecolor='#222222',
    fancybox=False,
    borderpad=0.6,
    handlelength=2.0,
    handletextpad=0.5,
    labelspacing=0.4,
)
legend.get_frame().set_linewidth(1.4)

# ── 8. Save ───────────────────────────────────────
os.makedirs(output_folder, exist_ok=True)
save_path = os.path.join(output_folder, output_filename)
plt.tight_layout()
plt.savefig(save_path, dpi=600, bbox_inches='tight', format='png')
print(f"\nFigure saved to:\n{save_path}")
plt.show()