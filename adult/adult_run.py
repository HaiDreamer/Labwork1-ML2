import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# -----------------------
# 1) Load data (online)
# -----------------------
cols = ["age","workclass","fnlwgt","education","education-num","marital-status",
        "occupation","relationship","race","sex","capital-gain","capital-loss",
        "hours-per-week","native-country","income"]

train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
test_url  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

df_train = pd.read_csv(train_url, names=cols, skipinitialspace=True)
df_test  = pd.read_csv(test_url,  names=cols, skiprows=1, skipinitialspace=True)  # skip header
df_test["income"] = df_test["income"].str.replace(".", "", regex=False)           # strip trailing '.'

df = pd.concat([df_train, df_test], ignore_index=True)

# -----------------------
# 2) Clean missing '?'
# -----------------------
df = df.replace("?", np.nan)

print("Missing counts (top 10):")
print(df.isna().sum().sort_values(ascending=False).head(10))

rows_before = len(df)
df = df.dropna().copy()
rows_after = len(df)
print(f"\nRows before dropna: {rows_before}")
print(f"Rows after dropna : {rows_after}")
print(f"Rows removed      : {rows_before - rows_after}")

dup_before = df.duplicated().sum()
df = df.drop_duplicates().copy()
dup_after = df.duplicated().sum()
print(f"\nExact duplicate rows removed: {dup_before - dup_after}")

# -----------------------
# 3) Split X/y
# -----------------------
y = df["income"].astype(str)
X = df.drop(columns=["income"])

num_cols = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
cat_cols = [c for c in X.columns if c not in num_cols]

# -----------------------
# 4) Stats for numeric features
# -----------------------
num = X[num_cols].astype(float)
mean = num.mean()
var = num.var(ddof=1)
cov = num.cov()
corr = num.corr()

print("\nMean (numeric):")
print(mean.round(4))
print("\nVariance (numeric):")
print(var.round(4))

corr_abs = corr.abs().copy()
np.fill_diagonal(corr_abs.values, 0)
i, j = np.unravel_index(np.argmax(corr_abs.values), corr_abs.shape)
print("\nMost correlated numeric pair:")
print(f"{corr.index[i]} & {corr.columns[j]}  r = {corr.iloc[i, j]:.4f}")

def heatmap(mat, labels, filename, title):
    plt.figure(figsize=(7,6))
    plt.imshow(mat.values, aspect="auto")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

heatmap(cov,  num_cols, "adult_covariance.png",  "Adult: Covariance (numeric)")
heatmap(corr, num_cols, "adult_correlation.png", "Adult: Correlation (numeric)")

# -----------------------
# 5) PCA: one-hot + standardize + PCA
# -----------------------
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", ohe, cat_cols)
    ]
)

pca = PCA(random_state=0)
pipe = Pipeline([("prep", preprocess), ("pca", pca)])

Z = pipe.fit_transform(X)

evr = pipe.named_steps["pca"].explained_variance_ratio_
cum = np.cumsum(evr)

print("\nEVR first 10:", np.round(evr[:10], 4))
print("CumEVR first 10:", np.round(cum[:10], 4))

k95 = int(np.argmax(cum >= 0.95) + 1)
print("\n#PCs for 95% variance:", k95)

# 2D scatter PC1 vs PC2
Z2 = Z[:, :2]
mask_hi = (y == ">50K")
plt.figure(figsize=(7,5))
plt.scatter(Z2[~mask_hi,0], Z2[~mask_hi,1], s=5, alpha=0.5, label="<=50K")
plt.scatter(Z2[mask_hi,0],  Z2[mask_hi,1],  s=5, alpha=0.5, label=">50K")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig("adult_2Dvisual.png", dpi=200)
plt.close()

print("\nSaved images: adult_covariance.png, adult_correlation.png, adult_2Dvisual.png")
