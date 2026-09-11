import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# ─────────────────────────────────────────────────────────────
# STYLE SETTINGS
# ─────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 1.5

# Make math text (e.g. PM$_{2.5}$) render in a Times-like serif font too,
# so it matches the rest of the labels instead of falling back to the
# default STIX/DejaVu math font.
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# File path
file_path   = r"C:\Users\ocean\OneDrive\Desktop\OA Thesis\Standarized Data\Plot season Pm2.5.xlsx"
output_path = r"C:\Users\ocean\OneDrive\Desktop\OA Thesis\Python5"

# Read Excel file
df = pd.read_excel(file_path)

# Reshape data to long format
df_long = pd.melt(df, var_name='Season', value_name='PM2.5')
df_long = df_long.dropna()

# Define season order
season_order = ['Winter', 'Premonsoon', 'Monsoon', 'Postmonsoon']

# Trim min-max
def trim_min_max(group):
    min_val = group['PM2.5'].min()
    max_val = group['PM2.5'].max()
    return group[(group['PM2.5'] >= min_val) & (group['PM2.5'] <= max_val)]

# Suppress the deprecation warning
warnings.filterwarnings('ignore', category=DeprecationWarning)
df_long = df_long.groupby('Season', group_keys=False).apply(trim_min_max)

# ---------------------------------------------------
# Plot
# ---------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Fix for FutureWarning: add hue='Season' and legend=False
sns.boxplot(
    data=df_long,
    x='Season',
    y='PM2.5',
    hue='Season',
    order=season_order,
    palette=['#5DADE2', '#52BE80', '#F39C12', '#E74C3C'],
    width=0.6,
    linewidth=2,
    showfliers=False,
    legend=False,
    ax=ax
)

# Guidelines
ax.axhline(y=15, color='red', linestyle='--', linewidth=3, label='WHO 24-hr Guideline (15 µg/m³)')
ax.axhline(y=40, color='darkorange', linestyle='--', linewidth=3, label='Nepal 24-hr Standard (40 µg/m³)')

# ============================================================
# AXIS LABELS — sizes matched to Figures 1 & 2 (axis titles = 24, labelpad = 10)
# ============================================================
ax.set_xlabel('Season', fontsize=24, fontweight='bold', labelpad=10,
              fontfamily='Times New Roman')
ax.set_ylabel(r'$\mathbf{PM_{2.5}}$ $\mathbf{Concentration}$ $\mathbf{(\mu g/m^3)}$',
              fontsize=24, fontweight='bold', labelpad=10,
              fontfamily='Times New Roman')

# ============================================================
# TICKS — sizes matched to Figures 1 & 2 (tick numbers = 16)
# ============================================================
ax.tick_params(axis='both', which='major', labelsize=16, width=1.5, length=6)

# Force Times New Roman on every tick label — tick_params alone
# doesn't set font family.
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontfamily('Times New Roman')

# ============================================================
# LEGEND — sizes/format matched to Figures 1 & 2 (legend = 20)
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
ax.grid(True, axis='y', alpha=0.4, linestyle='--', linewidth=0.8, color='gray')
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_edgecolor('#333333')
    spine.set_linewidth(1.5)

ax.set_ylim(0, None)

plt.tight_layout()

# ============================================================
# SAVE & SHOW
# ============================================================
filename  = "Seasonal_PM25_BoxPlot.png"
save_path = f"{output_path}\\{filename}"
plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
print(f"\nSaved -> {save_path}")

plt.show()
plt.close()