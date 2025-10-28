# 🤖 Clustering Using EM Algorithm and K-Means

## 📌 Objective
This project demonstrates how to perform **clustering** using:
1. **Expectation-Maximization (EM)** — implemented via Gaussian Mixture Models (GMM)
2. **K-Means Clustering**  
Both are applied to the same dataset stored in a `.csv` file.

---

## ⚙️ Step 1: Import Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Load your dataset (use the generated clustering_data.csv)
df = pd.read_csv("clustering_data.csv")
print("✅ Dataset Loaded Successfully!\n")
print(df.head())

plt.scatter(df['Feature_1'], df['Feature_2'], s=30, c='gray')
plt.title("Raw Data (Unclustered)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Initialize and fit KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(df)

# Plot K-Means Clusters
plt.scatter(df['Feature_1'], df['Feature_2'], c=df['KMeans_Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='red', marker='X', s=200)
plt.title("📊 K-Means Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Initialize Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(df[['Feature_1', 'Feature_2']])
df['EM_Cluster'] = gmm.predict(df[['Feature_1', 'Feature_2']])

# Plot EM Clusters
plt.scatter(df['Feature_1'], df['Feature_2'], c=df['EM_Cluster'], cmap='coolwarm')
plt.title("🤖 EM Algorithm (Gaussian Mixture) Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(df['Feature_1'], df['Feature_2'], c=df['KMeans_Cluster'], cmap='viridis')
axes[0].set_title("K-Means Clustering")
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")

axes[1].scatter(df['Feature_1'], df['Feature_2'], c=df['EM_Cluster'], cmap='coolwarm')
axes[1].set_title("EM Algorithm (GMM) Clustering")
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")

plt.tight_layout()
plt.show()

```
