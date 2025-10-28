# 🤖 Linear Regression & Logistic Regression Implementation

This project demonstrates the implementation of **Linear Regression** and **Logistic Regression** using Python’s powerful `scikit-learn` library.  
These are fundamental machine learning algorithms used for **prediction** and **classification** tasks.

---

## ⚙️ Installation

Make sure to install the required libraries before running the code:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```
## 📈 Linear Regression

### 🔍 What is Linear Regression?

**Linear Regression** is a supervised machine learning algorithm used to **predict a continuous dependent variable** (e.g., price, salary, sales) based on one or more independent variables.  
It tries to find the **best-fit line (y = mx + c)** that represents the relationship between variables.

---

### 🧠 Example: Predicting Sales Based on Advertising Budget

```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create example dataset
data = {
    'Advertising_Spend': [50, 60, 70, 80, 90, 100, 110, 120],
    'Sales': [150, 160, 180, 200, 210, 220, 230, 250]
}
df = pd.DataFrame(data)

# Define independent (X) and dependent (y) variables
X = df[['Advertising_Spend']]
y = df['Sales']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
print("✅ Linear Regression Results:")
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Visualization
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.title('📈 Linear Regression: Advertising vs Sales')
plt.xlabel('Advertising Spend')
plt.ylabel('Sales')
plt.legend()
plt.show()
```

## 📊 Logistic Regression

### 🔍 What is Logistic Regression?

**Logistic Regression** is a **classification algorithm** used to predict **binary outcomes** (e.g., Yes/No, Pass/Fail, 0/1).  
It estimates the **probability of an event occurring** using the **sigmoid (logistic) function**.

---

### 🧠 Example: Predicting if a Student Passes an Exam Based on Study Hours

```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Create sample dataset
data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Pass': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# Split data into features and target
X = df[['Study_Hours']]
y = df['Pass']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
print("✅ Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualization — Logistic curve
X_range = np.linspace(0, 10, 100).reshape(-1, 1)
y_prob = model.predict_proba(X_range)[:, 1]

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_range, y_prob, color='red', label='Logistic Curve')
plt.title('📊 Logistic Regression: Study Hours vs Pass Probability')
plt.xlabel('Study Hours')
plt.ylabel('Probability of Passing')
plt.legend()
plt.show()

# Heatmap of confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```
