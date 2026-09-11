from datetime import datetime
import os
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────
# STYLE SETTINGS
# ─────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 1.5

# Make math text render in Times New Roman
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# Read the data
file_path = r'C:\Users\ocean\OneDrive\Desktop\OA Thesis\Standarized Data\realtionship\all_houly.csv'
df = pd.read_csv(file_path)

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

# Extract time components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['year_month'] = df['date'].dt.to_period('M')

# Calculate monthly averages
monthly_data = (
    df.groupby('year_month')
    .agg({'Pm2.5': 'mean', 'month': 'first', 'year': 'first'})
    .reset_index()
)

# Convert year_month back to datetime for plotting
monthly_data['date'] = monthly_data['year_month'].dt.to_timestamp()


# Define seasons
def assign_season(month):
  if month in [12, 1, 2]:
    return 'Winter'
  elif month in [3, 4, 5]:
    return 'Pre-monsoon'
  elif month in [6, 7, 8, 9]:
    return 'Monsoon'
  elif month in [10, 11]:
    return 'Post-monsoon'


monthly_data['season'] = monthly_data['month'].apply(assign_season)

# Define season colors
season_colors = {
    'Winter': '#BBDEFB',
    'Pre-monsoon': '#FFE0B2',
    'Monsoon': '#C8E6C9',
    'Post-monsoon': '#F8BBD0',
}

# Output path
output_path = r'C:\Users\ocean\OneDrive\Desktop\OA Thesis\Python5'

# ============================================================
# CREATE TIME SERIES PLOT
# ============================================================
fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor('white')

# Darker background for the plot area
ax.set_facecolor('#E0E0E0')

# Seasonal shading
current_season = None
season_start = monthly_data['date'].iloc[0]

for i, row in monthly_data.iterrows():
  if row['season'] != current_season:
    if current_season is not None:
      ax.axvspan(
          season_start,
          row['date'],
          alpha=0.4,
          color=season_colors[current_season],
          zorder=1,
      )
    current_season = row['season']
    season_start = row['date']

# Last season span
if current_season is not None:
  ax.axvspan(
      season_start,
      monthly_data['date'].iloc[-1],
      alpha=0.4,
      color=season_colors[current_season],
      zorder=1,
  )

# Guidelines
ax.axhline(
    y=15,
    color='#D32F2F',
    linestyle='--',
    linewidth=2,
    label='WHO 24-hr Guideline (15 µg/m³)',
    zorder=4,
)
ax.axhline(
    y=40,
    color='#E65100',
    linestyle='--',
    linewidth=2,
    label='Nepal 24-hr Standard (40 µg/m³)',
    zorder=4,
)

# PM2.5 Line
ax.plot(
    monthly_data['date'],
    monthly_data['Pm2.5'],
    color='#0D47A1',
    linewidth=2.5,
    marker='o',
    markersize=5,
    label='Monthly Average PM$_{2.5}$',
    zorder=5,
)

# ============================================================
# HIGHLIGHT MARCH–APRIL 2021 WILDFIRE SURGE
# Star only on the peak — label goes in the legend (no separate box)
# ============================================================
fire_mask = (monthly_data['date'] >= '2021-03-01') & (
    monthly_data['date'] <= '2021-04-30'
)
if fire_mask.any():
  fire_peak_row = monthly_data.loc[monthly_data[fire_mask]['Pm2.5'].idxmax()]
  fire_date = fire_peak_row['date']
  fire_val = fire_peak_row['Pm2.5']
else:
  fire_date = pd.to_datetime('2021-03-01')
  fire_val = monthly_data['Pm2.5'].max()

# Red star at the peak only (no annotate box / arrow)
ax.plot(
    fire_date,
    fire_val,
    marker='*',
    markersize=18,
    color='#D50000',
    markeredgecolor='black',
    markeredgewidth=1.0,
    zorder=6,
    label='_nolegend_',  # legend entry added manually below
)

# ============================================================
# AXIS LABELS
# ============================================================
ax.set_xlabel(
    'Year', fontsize=24, fontweight='bold', labelpad=10, fontfamily='Times New Roman'
)
ax.set_ylabel(
    r'$\mathbf{PM_{2.5}}$ $\mathbf{Concentration}$ $\mathbf{(\mu g/m^3)}$',
    fontsize=24,
    fontweight='bold',
    labelpad=10,
    fontfamily='Times New Roman',
)

# ============================================================
# TICKS
# ============================================================
ax.tick_params(axis='both', which='major', labelsize=16, width=1.5, length=6)

for label in ax.get_xticklabels() + ax.get_yticklabels():
  label.set_fontfamily('Times New Roman')

# Custom legend patches (seasons)
winter_p = mpatches.Patch(
    color=season_colors['Winter'], alpha=0.6, label='Winter (Dec-Feb)'
)
pre_p = mpatches.Patch(
    color=season_colors['Pre-monsoon'],
    alpha=0.6,
    label='Pre-monsoon (Mar-May)',
)
mon_p = mpatches.Patch(
    color=season_colors['Monsoon'], alpha=0.6, label='Monsoon (Jun-Sep)'
)
post_p = mpatches.Patch(
    color=season_colors['Post-monsoon'],
    alpha=0.6,
    label='Post-monsoon (Oct-Nov)',
)

# Forest fire entry for legend (star symbol + text) — placed after Post-monsoon
fire_legend = Line2D(
    [0], [0],
    marker='*',
    color='none',
    markerfacecolor='#D50000',
    markeredgecolor='black',
    markeredgewidth=1.0,
    markersize=16,
    label='Forest Fire Surge (Mar–Apr 2021)',
)

# Combine handles: line handles first, then seasons, then fire star at the end
handles, labels = ax.get_legend_handles_labels()
handles.extend([winter_p, pre_p, mon_p, post_p, fire_legend])

# ============================================================
# LEGEND (top right — fire label sits below Post-monsoon inside the box)
# ============================================================
legend = ax.legend(
    handles=handles,
    loc='upper right',
    prop={'family': 'Times New Roman', 'size': 16},
    frameon=True,
    facecolor='white',
    framealpha=1,
    edgecolor='black',
    bbox_to_anchor=(0.999, 0.999),
    bbox_transform=ax.transAxes,
)
legend.get_frame().set_linewidth(1.5)
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_facecolor('white')

# ============================================================
# GRID & BORDERS
# ============================================================
ax.grid(True, alpha=0.5, linestyle=':', color='white', zorder=2)

for spine in ax.spines.values():
  spine.set_edgecolor('#333333')
  spine.set_linewidth(1.5)

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Headroom for legend
ax.set_ylim(0, monthly_data['Pm2.5'].max() * 1.35)

plt.tight_layout()

# ============================================================
# SAVE & SHOW
# ============================================================
os.makedirs(output_path, exist_ok=True)
filename = 'Monthly_PM25_TimeSeries_Dark.png'
save_path = os.path.join(output_path, filename)
plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
print(f'\nSaved -> {save_path}')

plt.show()
plt.close()