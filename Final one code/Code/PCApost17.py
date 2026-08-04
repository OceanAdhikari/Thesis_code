import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os

# ── Varimax rotation function ──────────────────────────────────────────────
def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for _ in range(q):
        d_old = d
        Lambda = Phi @ R
        u, s, vh = np.linalg.svd(
            Phi.T @ (Lambda**3 - (gamma/p) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda)))
        )
        R = u @ vh
        d = np.sum(s)
        if d_old != 0 and d/d_old < 1 + tol:
            break
    return Phi @ R
# ──────────────────────────────────────────────────────────────────────────

# Folder path and confirmed CSV name for POSTMONSOON
folder_path = r'C:\Users\ocean\OneDrive\Desktop\OA Thesis\Standarized Data\Postmonsoon\daily\Dailystandarized'
csv_file_name = 'dailystandarized.csv'
file_path = os.path.join(folder_path, csv_file_name)

# Load data (already standardized — no need to standardize again)
data = pd.read_csv(file_path)

print("Data loaded successfully!")
print("Columns:", data.columns.tolist())
print("Shape:", data.shape)

# Perform PCA
pca = PCA()
pca.fit(data)

# Eigenvalues and variance
eigenvalues = pca.explained_variance_
var_percent = pca.explained_variance_ratio_ * 100
cum_eigen = np.cumsum(eigenvalues)
cum_percent = np.cumsum(var_percent)

# Table 3 — unchanged
table3 = pd.DataFrame({
    'Factor No.': [f'PC{i+1}' for i in range(len(eigenvalues))],
    'Eigenvalue': eigenvalues.round(4),
    '% of Total Variance': var_percent.round(2),
    'Cumul. Eigenvalue': cum_eigen.round(4),
    'Cumul.%': cum_percent.round(2)
})

print("\n=== Table 3: Eigenvalues of the correlation matrix (Postmonsoon season) ===")
print(table3)

# ── Apply Varimax to the loadings ─────────────────────────────────────────
# Select PCs with eigenvalue > 0.8
n_pcs = np.sum(eigenvalues > 0.8)
print(f"\nNumber of components with eigenvalue > 0.8: {n_pcs}")

# Get unrotated loadings
unrotated_loadings = pca.components_.T[:, :n_pcs]

# Apply Varimax rotation
rotated_loadings = varimax(unrotated_loadings)

# Table 4 — Varimax rotated loadings
loadings_df = pd.DataFrame(
    rotated_loadings,
    index=data.columns,
    columns=[f'RC{i+1}' for i in range(n_pcs)]  # RC = Rotated Component
).round(3)

print("\n=== Table 4: Varimax Rotated Factor Loadings (Postmonsoon season) ===")
print(loadings_df)
# ──────────────────────────────────────────────────────────────────────────

# Save as CSV files
table3_path = os.path.join(folder_path, 'PCA_Table3_Postmonsoon.csv')
table3.to_csv(table3_path, index=False)
print(f"\nTable 3 saved to: {table3_path}")

table4_path = os.path.join(folder_path, 'PCA_Table4_Varimax_Postmonsoon.csv')
loadings_df.to_csv(table4_path)
print(f"Table 4 saved to: {table4_path}")

print("\nAll done!")