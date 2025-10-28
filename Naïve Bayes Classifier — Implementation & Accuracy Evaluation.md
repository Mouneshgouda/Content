## 🤖 Naïve Bayes Classifier — Implementation & Accuracy Evaluation

### 🔍 What is Naïve Bayes?

**Naïve Bayes** is a **probabilistic machine learning algorithm** based on **Bayes’ Theorem**, which assumes that the predictors (features) are **independent** of each other given the class label.  

It is widely used for **classification problems** such as:
- Spam detection  
- Sentiment analysis  
- Medical diagnosis  
- Document categorization  

---

### 🧠 Example: Classifying Iris Flowers Using Naïve Bayes

We’ll use the **Iris dataset** — a famous dataset with 150 samples of iris flowers, classified into 3 species based on petal and sepal measurements.

---

```python
# ----------------------------------------------------------
# 📦 Import Required Libraries
# ----------------------------------------------------------
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 📂 Load Dataset
# ----------------------------------------------------------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print("✅ Dataset Loaded Successfully!")
print(df.head())

# ----------------------------------------------------------
# 🔢 Split Data into Features and Target
# ----------------------------------------------------------
X = df.drop('species', axis=1)
y = df['species']

# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------------------------
# 🧠 Train Naïve Bayes Model
# ----------------------------------------------------------
model = GaussianNB()
model.fit(X_train, y_train)

# ----------------------------------------------------------
# 🔍 Make Predictions
# ----------------------------------------------------------
y_pred = model.predict(X_test)

# ----------------------------------------------------------
# 📈 Evaluate Model
# ----------------------------------------------------------
print("\n✅ Naïve Bayes Classification Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------------------------------------
# 📊 Visualization — Confusion Matrix
# ----------------------------------------------------------
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix — Naïve Bayes Classifier (Iris Dataset)')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
