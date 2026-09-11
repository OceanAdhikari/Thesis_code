import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.4)

# Set Times New Roman font globally — done AFTER plt.style.use()/sns.set_context(),
# since either of those can silently reset font.family back to the style default.
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 1.5
# Make math text (e.g. PM$_{2.5}$) render in a Times-like serif font too,
# so it matches the rest of the labels instead of falling back to the
# default STIX/DejaVu math font.
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# ============================================================
# FILE PATH & DATA LOADING
# ============================================================
file_path   = r"C:\Users\ocean\OneDrive\Desktop\OA Thesis\High Pollution cases\Precipitation\Again\Merged_PM25_Precipitation.csv"
output_path = r"C:\Users\ocean\OneDrive\Desktop\OA Thesis\Python5"

df = pd.read_csv(file_path)

print("Columns:", df.columns.tolist())
print("Shape  :", df.shape)
print(df.head())

# ============================================================
# COLUMN NAMES
# ============================================================
PM_COL     = 'PM2.5_daily'
PREC_COL   = 'Precipitation'
DATE_COL   = 'date'
SEASON_COL = 'season'

# ============================================================
# DATE PARSING & SEASON ASSIGNMENT
# ============================================================
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df['month']  = df[DATE_COL].dt.month

def assign_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Pre-monsoon'
    elif month in [6, 7, 8, 9]:
        return 'Monsoon'
    elif month in [10, 11]:
        return 'Post-monsoon'

df[SEASON_COL] = df['month'].apply(assign_season)

season_colors = {
    'Winter':       '#2E86AB',
    'Pre-monsoon':  '#F77F00',
    'Monsoon':      '#06A77D',
    'Post-monsoon': '#D62828'
}

# ============================================================
# OUTLIER REMOVAL
# - PM2.5      : IQR method (works fine, roughly symmetric)
# - Precipitation: 99th percentile cap because most values = 0
#   so IQR gives a tiny upper bound (~7 mm) and wrongly removes
#   genuine heavy rain events
# ============================================================
def remove_outliers_pm(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    print(f"  [PM2.5  IQR bounds ] {lower:.2f} - {upper:.2f} ug/m3")
    return data[(data[col] >= lower) & (data[col] <= upper)]

def remove_outliers_precip(data, col, percentile=99):
    upper = data[col].quantile(percentile / 100)
    print(f"  [Precip {percentile}th pct cap] upper bound: {upper:.2f} mm")
    return data[data[col] <= upper]

# ============================================================
# PREPARE PLOT DATA
# ============================================================
plot_data = df[[PREC_COL, PM_COL, SEASON_COL]].dropna()
print(f"\nData points before outlier removal : {len(plot_data)}")
print("Outlier removal details:")

plot_data_clean = remove_outliers_pm(plot_data, PM_COL)
plot_data_clean = remove_outliers_precip(plot_data_clean, PREC_COL, percentile=99)

print(f"\nData points after  outlier removal : {len(plot_data_clean)}")
removed = len(plot_data) - len(plot_data_clean)
print(f"Total outliers removed             : {removed} ({removed/len(plot_data)*100:.1f}%)")

x = plot_data_clean[PREC_COL].values
y = plot_data_clean[PM_COL].values

print(f"\nPrecipitation range : {x.min():.3f} - {x.max():.3f} mm")
print(f"PM2.5 range         : {y.min():.2f} - {y.max():.2f} ug/m3")

# ============================================================
# LINEAR REGRESSION
# ============================================================
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(f"\nCorrelation  r  : {r_value:.4f}")
print(f"R2              : {r_value**2:.4f}")
print(f"p-value         : {p_value:.6f}")

# ============================================================
# FIGURE
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# --- scatter by season ---
season_order = ['Winter', 'Pre-monsoon', 'Monsoon', 'Post-monsoon']
for season in season_order:
    sd = plot_data_clean[plot_data_clean[SEASON_COL] == season]
    if len(sd) > 0:
        ax.scatter(sd[PREC_COL], sd[PM_COL],
                   c=season_colors[season], label=season,
                   alpha=0.70, s=45, edgecolors='white', linewidth=0.4,
                   zorder=3)

# --- trend line clipped at y=0 ---
x_range      = x.max() - x.min()
x_line       = np.linspace(0, x.max() + x_range * 0.05, 300)
y_line       = slope * x_line + intercept
y_line       = np.clip(y_line, 0, None)   # prevent negative PM2.5

ax.plot(x_line, y_line,
        color='black', linewidth=2.5, linestyle=':',
        dashes=(2, 3), label='Trend Line', alpha=0.9, zorder=4)

# --- r annotation ---
ax.text(0.10, 0.97, f'r = {r_value:.3f}',
        transform=ax.transAxes, fontsize=22,
        verticalalignment='top', horizontalalignment='left',
        fontweight='bold', color='black', fontfamily='Times New Roman')

# ============================================================
# AXIS LABELS
# ============================================================
ax.set_xlabel('Daily Precipitation (mm)',
              fontsize=26, fontweight='bold', labelpad=12,
              fontfamily='Times New Roman')
ax.set_ylabel(r'PM$_{2.5}$ Concentration ($\mu$g/m$^3$)',
              fontsize=26, fontweight='bold', labelpad=12,
              fontfamily='Times New Roman')

# ============================================================
# X-AXIS TICKS — based on actual cleaned data max
# ============================================================
x_data_max = x.max()
x_axis_max = np.ceil(x_data_max * 1.08)

ax.set_xlim(-x_data_max * 0.015, x_axis_max)

# Dynamic tick step based on actual max
if x_data_max <= 10:
    tick_step = 1
elif x_data_max <= 25:
    tick_step = 5
elif x_data_max <= 60:
    tick_step = 5
elif x_data_max <= 100:
    tick_step = 10
elif x_data_max <= 200:
    tick_step = 20
else:
    tick_step = 25

major_ticks = np.arange(0, x_axis_max + tick_step, tick_step)
ax.set_xticks(major_ticks)

ax.minorticks_on()
ax.tick_params(axis='x', which='minor', length=4, width=1, color='#aaaaaa')
ax.tick_params(axis='both', which='major', labelsize=20, width=1.5, length=7)

# ============================================================
# Y-AXIS RANGE
# ============================================================
y_upper = y.max() + (y.max() - y.min()) * 0.32
ax.set_ylim(0, y_upper)
ax.tick_params(axis='y', which='major', labelsize=20, width=1.5, length=7)

# Force Times New Roman + BOLD on every tick label (x and y, major and minor),
# since tick_params alone doesn't set font family or weight.
for label in ax.get_xticklabels() + ax.get_yticklabels() + \
             ax.get_xticklabels(minor=True) + ax.get_yticklabels(minor=True):
    label.set_fontfamily('Times New Roman')
    label.set_fontweight('bold')

# ============================================================
# LEGEND
# ============================================================
legend = ax.legend(
    loc='upper right',
    prop={'family': 'Times New Roman', 'size': 20},
    frameon=True,
    facecolor='white',
    framealpha=1,
    edgecolor='black',
    bbox_to_anchor=(1.0, 1.0)
)
legend.get_frame().set_linewidth(1.5)
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_facecolor('white')

# ============================================================
# GRID & BORDERS
# ============================================================
ax.grid(True, which='major', alpha=0.35, linestyle='--',
        linewidth=0.8, color='gray')
ax.grid(False, which='minor')
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_edgecolor('#222222')
    spine.set_linewidth(1.5)

# ============================================================
# SAVE & SHOW
# ============================================================
plt.tight_layout(pad=1.5)
save_path = output_path + r"\PM25_vs_Precipitation.png"
plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
print(f"\nSaved -> {save_path}")
plt.show()
plt.close()