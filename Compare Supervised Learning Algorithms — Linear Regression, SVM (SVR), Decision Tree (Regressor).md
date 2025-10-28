# 📊 Compare Supervised Learning Algorithms — Linear Regression, SVM (SVR), Decision Tree (Regressor)

This project compares three **supervised learning algorithms** for a regression task:

- **Linear Regression**
- **Support Vector Machine (SVR)**
- **Decision Tree Regressor**

We generate a custom dataset, train each model, and compare their performances using evaluation metrics and visualizations.

---

## 🧠 Objective
To evaluate and compare the performance of **Linear Regression**, **Support Vector Machine (SVR)**, and **Decision Tree Regressor** using a synthetic dataset.

---

## 📦 Requirements

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```
## Data Overview & Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('supervised_compare.csv')

# Scatter plot
plt.scatter(df['x1'], df['y'], s=10, alpha=0.5)
plt.xlabel('x1'); plt.ylabel('y'); plt.title('y vs x1 (Nonlinear Relationship)')
plt.show()

# Boxplot by category
sns.boxplot(x='x3', y='y', data=df)
plt.title('Distribution of y by Category (x3)')
plt.show()

```

## Preprocessing and Model Training
```
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Split features and target
X = df.drop(columns=['y'])
y = df['y']

numeric_features = ['x1', 'x2', 'x4']
categorical_features = ['x3']

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Linear Regression': Pipeline([('prep', preprocessor), ('model', LinearRegression())]),
    'SVR (RBF)': Pipeline([('prep', preprocessor), ('model', SVR(kernel='rbf', C=10, epsilon=0.1))]),
    'Decision Tree': Pipeline([('prep', preprocessor), ('model', DecisionTreeRegressor(max_depth=6, random_state=42))])
}
```
## Evaluation Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

results = {}

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
    print(f"\n✅ {name}")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

```
## Cross-Validation & Visualization
```
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt

cv = KFold(n_splits=5, shuffle=True, random_state=42)

for name, pipeline in models.items():
    scores = cross_val_score(pipeline, X, y, scoring='r2', cv=cv)
    print(f"{name}: Mean R2 = {scores.mean():.4f}, Std = {scores.std():.4f}")

# Visualize predictions
import matplotlib.pyplot as plt

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    plt.scatter(y_test, y_pred, alpha=0.6, label=name)

plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.title("True vs Predicted Comparison")
plt.show()
```

