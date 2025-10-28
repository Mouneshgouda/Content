## 🌳 Decision Tree — ID3 Algorithm Implementation

### 🔍 What is the ID3 Algorithm?

**ID3 (Iterative Dichotomiser 3)** is a classic **Decision Tree algorithm** used for **classification problems**.  
It builds the tree top-down by selecting the attribute that gives the **highest information gain** at each node, based on **entropy**.

ID3 works best with **categorical data** and produces human-readable decision rules.

---

### ⚙️ Steps Involved
1. Load the dataset  
2. Compute **Entropy** and **Information Gain**  
3. Build a **Decision Tree** recursively using the ID3 algorithm  
4. Use the tree for prediction  
5. Evaluate accuracy  

---

### 🧠 Example: Classifying Whether a Person Buys a Computer

```python
# ----------------------------------------------------------
# 📦 Import Required Libraries
# ----------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 📂 Sample Dataset
# ----------------------------------------------------------
data = {
    'Age': ['<=30', '<=30', '31-40', '>40', '>40', '>40', '31-40', '<=30', '<=30', '>40', '<=30', '31-40', '31-40', '>40'],
    'Income': ['High', 'High', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium', 'Low', 'Medium', 'Medium', 'Medium', 'High', 'Medium'],
    'Student': ['No', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No'],
    'Credit_Rating': ['Fair', 'Excellent', 'Fair', 'Fair', 'Fair', 'Excellent', 'Excellent', 'Fair', 'Fair', 'Fair', 'Excellent', 'Excellent', 'Fair', 'Excellent'],
    'Buys_Computer': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)
print("✅ Dataset Loaded Successfully!")
print(df.head())

# ----------------------------------------------------------
# 🔢 Encode Categorical Data
# ----------------------------------------------------------
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Features and Target
X = df.drop('Buys_Computer', axis=1)
y = df['Buys_Computer']

# ----------------------------------------------------------
# ✂️ Split Data into Train and Test Sets
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----------------------------------------------------------
# 🌳 Train Decision Tree using ID3 Algorithm
# ----------------------------------------------------------
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# ----------------------------------------------------------
# 🔍 Make Predictions
# ----------------------------------------------------------
y_pred = model.predict(X_test)

# ----------------------------------------------------------
# 📈 Evaluate Model
# ----------------------------------------------------------
print("\n✅ Decision Tree (ID3) Classification Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------------------------------------
# 🌲 Visualize Decision Tree
# ----------------------------------------------------------
plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, rounded=True)
plt.title('🌳 Decision Tree using ID3 Algorithm')
plt.show()
```
