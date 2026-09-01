import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ── Varimax rotation function ──────────────────────────────────────────────
def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for _ in range(q):
        d_old = d
        Lambda = Phi @ R
        u, s, vh = np.linalg.svd(
            Phi.T
            @ (
                Lambda**3
                - (gamma / p) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda))
            )
        )
        R = u @ vh
        d = np.sum(s)
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return Phi @ R


# ──────────────────────────────────────────────────────────────────────────

# Folder path and CSV for WINTER
folder_path = r"C:\Users\ocean\OneDrive\Desktop\OA Thesis\Standarized Data\Winter\daily\Dailystandarization"
csv_file_name = "dailystandarized.csv"
file_path = os.path.join(folder_path, csv_file_name)

# Load data
data = pd.read_csv(file_path)
print("=" * 70)
print("WINTER: Data loaded successfully!")
print("Columns:", data.columns.tolist())
print("Shape  :", data.shape)
print("=" * 70)

# Perform PCA
pca = PCA()
pca.fit(data)

# Unrotated Eigenvalues and variance
eigenvalues = pca.explained_variance_
var_percent = pca.explained_variance_ratio_ * 100
cum_eigen = np.cumsum(eigenvalues)
cum_percent = np.cumsum(var_percent)

# Table: Unrotated Eigenvalues & Variance
table_unrotated = pd.DataFrame(
    {
        "Factor No.": [f"PC{i+1}" for i in range(len(eigenvalues))],
        "Eigenvalue": eigenvalues.round(4),
        "% of Total Variance": var_percent.round(2),
        "Cumul. Eigenvalue": cum_eigen.round(4),
        "Cumul.%": cum_percent.round(2),
    }
)

print("\n=== Table: Unrotated Eigenvalues & Variance (Winter) ===")
print(table_unrotated.to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────
# COMPONENT SELECTION & VARIMAX ROTATION
# ──────────────────────────────────────────────────────────────────────────
# Select PCs with eigenvalue > 0.8 (modified Kaiser criterion)
n_pcs = np.sum(eigenvalues > 0.8)
print(f"\nNumber of components retained (eigenvalue > 0.8): {n_pcs}")

# 1. Scale eigenvectors by sqrt(eigenvalues) to get TRUE unrotated loadings
unrotated_loadings = (
    pca.components_.T[:, :n_pcs] * np.sqrt(eigenvalues[:n_pcs])
)

# 2. Apply Varimax rotation to true loadings
rotated_loadings = varimax(unrotated_loadings)

# 3. Calculate variance explained after Varimax rotation
rotated_eigenvalues = np.sum(rotated_loadings**2, axis=0)
rotated_var_percent = (rotated_eigenvalues / len(data.columns)) * 100
rotated_cum_var_percent = np.cumsum(rotated_var_percent)

# Table: Varimax Rotated Factor Loadings
loadings_df = pd.DataFrame(
    rotated_loadings,
    index=data.columns,
    columns=[f"RC{i+1}" for i in range(n_pcs)],  # RC = Rotated Component
).round(3)

# Add variance rows at bottom
summary_rows = pd.DataFrame(
    {
        f"RC{i+1}": [
            round(rotated_eigenvalues[i], 4),
            round(rotated_var_percent[i], 2),
            round(rotated_cum_var_percent[i], 2),
        ]
        for i in range(n_pcs)
    },
    index=["Rotated Eigenvalue", "% Variance Explained", "Cumulative %"],
)

table_final = pd.concat([loadings_df, summary_rows])

print("\n=== Table: Varimax Rotated Factor Loadings & Variance (Winter) ===")
print(table_final)

# Save CSVs
table_unrotated.to_csv(
    os.path.join(folder_path, "PCA_Unrotated_Variance_Winter.csv"), index=False
)
table_final.to_csv(
    os.path.join(folder_path, "PCA_Varimax_Rotated_Loadings_Winter.csv")
)
print("\n[OK] Winter CSV files saved successfully!")