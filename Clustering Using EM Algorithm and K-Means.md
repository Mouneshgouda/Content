# 🔍 Clustering Using EM Algorithm and K-Means

## 🎯 Objective
The objective of this project is to **apply the Expectation-Maximization (EM) algorithm** for clustering a dataset and compare its results with the **K-Means algorithm** using the same data.  
Both algorithms will be evaluated based on their clustering effectiveness and visualization of the data points.

---

## 🧠 Learning Outcomes
After completing this exercise, students will be able to:
- Understand the concept of **unsupervised learning** and clustering.
- Implement **K-Means** and **EM (Gaussian Mixture Model)** algorithms using Python.
- Visualize cluster boundaries and analyze algorithm performance.
- Interpret results and compare EM and K-Means approaches.

---

## 🧩 Basic Concepts Required

| Concept | Description |
|----------|--------------|
| **K-Means Algorithm** | Partitions data into *k* clusters by minimizing the within-cluster variance. |
| **Expectation-Maximization (EM)** | Probabilistic model-based clustering algorithm using Gaussian Mixture Models (GMM). |
| **Clustering Evaluation** | Techniques to measure performance using metrics like silhouette score and log-likelihood. |
| **Standardization** | Scaling numerical data for better algorithm convergence. |

---

## 💾 About the Dataset
- **Name:** `clustering_data.csv`  
- **Attributes:** `Feature1`, `Feature2`  
- **Type:** Synthetic dataset generated for clustering tasks  
- **Description:**  
  The dataset contains 150 data points with two continuous features representing three natural clusters.

---

## 🧰 Software and Libraries

| Tool | Purpose |
|------|----------|
| **Python** | Programming language |
| **pandas** | Data handling |
| **numpy** | Numerical operations |
| **matplotlib / seaborn** | Visualization |
| **scikit-learn** | Machine learning algorithms (KMeans, GaussianMixture) |

---

## ⚙️ Tasks and Step-by-Step Approach

### 🧠 Task 1: Load and Explore Dataset
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('clustering_data.csv')
print(df.head())

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(df)

https://drive.google.com/file/d/1Nu0yeOTn43Gwb0VgGdnqmaoj7LT65br-/view?usp=sharing
```

## 💠 Task 2: K-Means Clustering
```python
from sklearn.cluster import KMeans

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Add cluster labels to dataset
df['KMeans_Cluster'] = kmeans_labels

# Visualize results
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('📊 K-Means Clustering')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.show()

https://drive.google.com/file/d/1NuPeN7P3DM-ByHidM3Uoen9IaW3WoRf0/view?usp=sharing
```
## 🔁 Task 3: EM Algorithm (Gaussian Mixture Model)
```python
from sklearn.mixture import GaussianMixture

# Apply EM algorithm
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(X)

df['EM_Cluster'] = gmm_labels

# Visualize EM results
plt.scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='coolwarm')
plt.title('🤖 EM (Gaussian Mixture) Clustering')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.show()

https://drive.google.com/file/d/1SZ7FIRufLQ8q-5XB22TSlW8l7O6_i10p/view?usp=sharing
```
## 📏 Task 4: Evaluation and Comparison
```python
from sklearn.metrics import silhouette_score

kmeans_score = silhouette_score(X, kmeans_labels)
gmm_score = silhouette_score(X, gmm_labels)

print("K-Means Silhouette Score:", kmeans_score)
print("EM (GMM) Silhouette Score:", gmm_score)

https://drive.google.com/file/d/1q3XgoTYk7qIhIBx7X6Cb347Op-J5QRzK/view?usp=sharing
```
## 🧾 Inputs

| Input | Description |
|--------|-------------|
| `clustering_data.csv` | Dataset containing numeric features for clustering |
| Python Script / Notebook | Implementation of K-Means and EM algorithms |
| Parameters | Number of clusters (default = 3) |

---

## 🎯 Expected Outputs

| Output | Description |
|--------|-------------|
| **Clustered Data** | Dataset with assigned cluster labels |
| **Visualizations** | Scatter plots showing clusters for K-Means and EM |
| **Evaluation Metrics** | Silhouette scores and comparison summary |

---

## 🔗 Google Colab Link

Run this project online using Google Colab:  
👉 [**Open in Google Colab**](https://colab.research.google.com/drive/1wmhVfJsGslEZSi5LOQKLILDhfFtWQDdX?usp=sharing)

*(Replace the link above with your actual Colab notebook URL once uploaded.)*

---

## 🧪 Testing and Validation

### ✅ 1. Test with Known Patterns
Use a simple dataset (like the one provided) to validate clustering visually.

**Expected Behavior:**
- Both algorithms should form **3 distinct clusters**.  
- EM may provide **softer boundaries** than K-Means.

---

### 📏 2. Validate Model Accuracy
- Compare **Silhouette Scores** between EM and K-Means.  
- Visualize **cluster separation** and **centroids**.  
- Adjust `n_clusters` to observe the impact on performance.

---

## 🧩 Troubleshooting Tips

| Issue | Likely Cause | Solution |
|--------|---------------|----------|
| `FileNotFoundError` | Dataset not found | Upload `clustering_data.csv` or correct file path. |
| `ValueError: could not convert string to float` | Non-numeric data in dataset | Ensure all features are numeric. |
| **Poor clustering** | Unscaled features or incorrect `k` value | Apply scaling and experiment with different `n_clusters`. |
| **Overlapping clusters** | High variance in features | Normalize data or reduce dimensions using PCA. |


## 💡 Example Projects Using Similar Approach

### 📊 Example 1: Customer Segmentation for Retail Business
**Objective:**  
Segment customers based on their shopping behavior to help a retail company identify high-value customers and personalize marketing campaigns.

**Dataset:**  
`Customer_Segmentation.csv` — contains attributes such as `Age`, `Annual_Income`, and `Spending_Score`.

**Possible Tasks:**
- Load and preprocess data.
- Apply **K-Means** and **EM (GMM)** to cluster customers.
- Visualize clusters based on income vs. spending score.
- Compare which algorithm forms more meaningful customer groups.

**Expected Output:**
- Clear separation of customer groups (e.g., low spenders, medium, high-value).  
- Insights on customer behavior patterns useful for business strategy.

---

### 🌦️ Example 2: Weather Pattern Clustering
**Objective:**  
Use EM and K-Means to identify weather types (e.g., sunny, cloudy, rainy) from meteorological data.

**Dataset:**  
`Weather_Data.csv` — includes features such as `Temperature`, `Humidity`, `Pressure`, and `WindSpeed`.

**Possible Tasks:**
- Standardize numeric features for clustering.
- Apply **K-Means** and **EM** for pattern detection.
- Visualize clusters on temperature vs. humidity scatterplots.
- Compare clustering consistency with known weather patterns.

**Expected Output:**
- Grouping of weather conditions into natural clusters.  
- EM provides smoother cluster boundaries than K-Means, useful for real-world variability.

---

### 🧠 Example 3: Image Color Compression (Optional Advanced Task)
**Objective:**  
Reduce image colors using clustering for compression and segmentation.

**Dataset:**  
Extracted RGB pixel values from an image file.

**Possible Tasks:**
- Convert image pixels into a dataset of RGB values.
- Apply **K-Means** and **EM** to cluster similar colors.
- Reconstruct the image using reduced color clusters.

**Expected Output:**
- Compressed image with fewer colors.  
- EM produces more realistic transitions, while K-Means is faster.

---

✅ *These examples follow the same structured workflow: load → preprocess → cluster → visualize → interpret.*
