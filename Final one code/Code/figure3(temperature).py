import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# ─────────────────────────────────────────────────────────────
# STYLE SETTINGS
# ─────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.4)

# Set Times New Roman font globally — done AFTER plt.style.use()/sns.set_context(),
# since either of those can silently reset font.family back to the style default.
plt.rcParams['font.family'] = 'Times New Roman'
# Make math text (e.g. PM$_{2.5}$) render in a Times-like serif font too,
# so it matches the rest of the labels instead of falling back to the
# default STIX/DejaVu math font.
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
file_path = r"C:\Users\ocean\OneDrive\Desktop\OA Thesis\Standarized Data\realtionship\all_houly.csv"
df = pd.read_csv(file_path)

df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

print("Original hourly data shape:", df.shape)
print("\nConverting hourly data to daily data...")

# ─────────────────────────────────────────────────────────────
# AGGREGATE TO DAILY DATA
# ─────────────────────────────────────────────────────────────
daily_data = df.groupby('date').agg({
    'Pm2.5': 'mean',
    'tmpf' : 'mean'
}).reset_index()

print(f"Daily data shape: {daily_data.shape}")

# ─────────────────────────────────────────────────────────────
# CONVERT FAHRENHEIT TO CELSIUS
# ─────────────────────────────────────────────────────────────
daily_data['tmpc'] = (daily_data['tmpf'] - 32) * 5 / 9

print(f"\nTemperature range:")
print(f"  Fahrenheit : {daily_data['tmpf'].min():.2f} to {daily_data['tmpf'].max():.2f} F")
print(f"  Celsius    : {daily_data['tmpc'].min():.2f} to {daily_data['tmpc'].max():.2f} C")

# ─────────────────────────────────────────────────────────────
# SEASON CLASSIFICATION
# ─────────────────────────────────────────────────────────────
daily_data['month'] = daily_data['date'].dt.month

def assign_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Pre-monsoon'
    elif month in [6, 7, 8, 9]:
        return 'Monsoon'
    elif month in [10, 11]:
        return 'Post-monsoon'

daily_data['season'] = daily_data['month'].apply(assign_season)

season_colors = {
    'Winter'       : '#2E86AB',
    'Pre-monsoon'  : '#F77F00',
    'Monsoon'      : '#06A77D',
    'Post-monsoon' : '#D62828'
}

# ─────────────────────────────────────────────────────────────
# OUTLIER REMOVAL (IQR)
# ─────────────────────────────────────────────────────────────
def remove_outliers(data, columns):
    df_clean = data.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

# ─────────────────────────────────────────────────────────────
# PLOT: PM2.5 vs TEMPERATURE (CELSIUS)
# ─────────────────────────────────────────────────────────────
output_path = r"C:\Users\ocean\OneDrive\Desktop\OA Thesis\Python5"
filename    = "PM25_vs_Temperature_Celsius.png"

print(f"\n{'='*80}")
print(f"Creating plot: {filename}")
print(f"{'='*80}")

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Prepare data
plot_data = daily_data[['tmpc', 'Pm2.5', 'season']].dropna()
print(f"Data points before outlier removal: {len(plot_data)}")

plot_data_clean = remove_outliers(plot_data, ['tmpc', 'Pm2.5'])
print(f"Data points after outlier removal : {len(plot_data_clean)}")
print(f"Outliers removed: {len(plot_data) - len(plot_data_clean)} "
      f"({((len(plot_data) - len(plot_data_clean))/len(plot_data)*100):.1f}%)")

x = plot_data_clean['tmpc'].values
y = plot_data_clean['Pm2.5'].values

print(f"Temperature range: {x.min():.2f} to {x.max():.2f} °C")
print(f"PM2.5       range: {y.min():.2f} to {y.max():.2f} µg/m³")

# Y-axis margin for legend
y_max_data = y.max()
y_margin   = (y.max() - y.min()) * 0.38
y_upper    = y_max_data + y_margin

# Plot by season
for season in ['Winter', 'Pre-monsoon', 'Monsoon', 'Post-monsoon']:
    season_data = plot_data_clean[plot_data_clean['season'] == season]
    if len(season_data) > 0:
        ax.scatter(season_data['tmpc'], season_data['Pm2.5'],
                   c=season_colors[season], label=season,
                   alpha=0.65, s=40,
                   edgecolors='white', linewidth=0.5)

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print(f"\nCorrelation (r): {r_value:.4f}")
print(f"R-squared      : {r_value**2:.4f}")
print(f"P-value        : {p_value:.6f}")

# Trend line
x_line = np.array([x.min(), x.max()])
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, color='black', linewidth=2.5,
        label='Trend Line', alpha=0.9,
        linestyle=':', dashes=(2, 3))

# Correlation text (top-left)
textstr = f'r = {r_value:.3f}'
ax.text(0.10, 0.97, textstr, transform=ax.transAxes,
        fontsize=22, verticalalignment='top',
        horizontalalignment='left',
        fontweight='bold', color='black', fontfamily='Times New Roman')

# Labels — sizes matched to Figures 1 & 2 (axis titles = 24)
ax.set_xlabel('Temperature (°C)',
              fontsize=26, fontweight='bold', labelpad=10,
              fontfamily='Times New Roman')
ax.set_ylabel(r'PM$_{2.5}$ Concentration ($\mu$g/m$^3$)',
              fontsize=26, fontweight='bold', labelpad=10,
              fontfamily='Times New Roman')

# Legend (top-right) — size matched to Figures 1 & 2 (legend = 20)
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

# Grid & background
ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8, color='gray')
ax.set_axisbelow(True)
ax.set_facecolor('#FAFAFA')

# Axis limits
x_margin = (x.max() - x.min()) * 0.10
ax.set_xlim(x.min() - x_margin*0.5, x.max() + x_margin)
ax.set_ylim(0, y_upper)

# Ticks — size matched to Figures 1 & 2 (tick numbers = 16)
ax.tick_params(axis='both', which='major', labelsize=20,
               width=1.5, length=6)

# Force Times New Roman on every tick label — tick_params alone
# doesn't set font family.
for label in ax.get_xticklabels() + ax.get_yticklabels() + \
             ax.get_xticklabels(minor=True) + ax.get_yticklabels(minor=True):
    label.set_fontfamily('Times New Roman')
    label.set_fontweight('bold')

# Borders
for spine in ax.spines.values():
    spine.set_edgecolor('#333333')
    spine.set_linewidth(1.5)

plt.tight_layout()

# Save
save_path = f"{output_path}\\{filename}"
plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
print(f"\n[OK] Saved: {filename}")
plt.show()
plt.close()

print("\n" + "="*80)
print("PLOT GENERATED SUCCESSFULLY!")
print("="*80)