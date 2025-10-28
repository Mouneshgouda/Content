# 🌸 K-Nearest Neighbors (KNN) Classification — Iris Dataset

## 🔍 What is KNN?
**K-Nearest Neighbors (KNN)** is a **supervised machine learning algorithm** used for **classification and regression**.  
It works by finding the **K nearest data points** in the feature space and assigning the most common class among them.

---

## ⚙️ Step 1: Import Required Libraries
```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load built-in iris dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("✅ Iris Dataset Loaded Successfully!\n")
print(df.head())

# Features (X) and Labels (y)
X = df.iloc[:, :-2].values
y = df['target'].values

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize KNN Classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Predict test set results
y_pred = knn.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ KNN Model Accuracy: {accuracy * 100:.2f}%\n")

# Confusion Matrix
print("📈 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Check accuracy for different k values
k_values = range(1, 11)
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_k = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred_k))

# Plot accuracy vs k
import matplotlib.pyplot as plt

plt.plot(k_values, scores, marker='o')
plt.title('K Value vs Accuracy')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
```
