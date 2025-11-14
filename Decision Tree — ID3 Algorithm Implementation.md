# Decision Tree Classifier using ID3 Algorithm

## Objective
This project demonstrates the implementation of a Decision Tree Classifier using the ID3 (Iterative Dichotomiser 3) algorithm.  
The model uses entropy and information gain to build a decision tree, enabling interpretable, rule-based decisions for classification problems.

---

## Learning Outcomes
By completing this project, you will be able to:

- Understand entropy information gain and decision tree construction.
- Build a decision tree using the ID3 algorithm
- Visualize and interpret decision-making in the tree.
- Evaluate model performance using metrics such as accuracy, confusion matrix, and classification report.
- Apply decision tree reasoning to other classification tasks.

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| Decision Tree | A supervised learning model that splits data based on features to make predictions. |
| ID3 Algorithm | Builds a decision tree by choosing splits that maximize information gain. |
| Entropy | Measures impurity or randomness in data. |
| Information Gain | Reduction in entropy after splitting data on a feature. |

---

## Dataset

File: `decision_tree_dataset.csv`  
Records: 200 samples  

| Feature | Description |
|---------|-------------|
| Age_Group | Categorical: Young / Middle-aged / Senior |
| Income_Level | Categorical: Low / Medium / High |
| Owns_House | Categorical: Yes / No |
| Marital_Status | Categorical: Single / Married |
| Education | Categorical: High School / Graduate / Post-Graduate |
| Purchased | Target variable: 1 = Purchased, 0 = Not Purchased |

> The dataset can be generated with a Python script or replaced with your own CSV file.

---

## Libraries Required

| Library | Purpose |
|---------|---------|
| Python 3.x | Programming language |
| pandas | Data handling |
| numpy | Numerical operations |
| scikit-learn | Decision Tree implementation and metrics |
| matplotlib | Visualization |
| seaborn | Enhanced plots |

---

## Step-by-Step Approach

### 1. Load and Explore Dataset
```python
import pandas as pd

df = pd.read_csv('decision_tree_dataset.csv')
print(df.head())
print(df.info())
print(df.describe())
```
### 2. Split Data into Features and Target
```c
X = df.drop('Purchased', axis=1)
y = df['Purchased']
```
### 3. Split Dataset into Training and Testing Sets
```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
### 4. Train Decision Tree Model using ID3
```c
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)
```
### 5. Evaluate Model
```c
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```
### 6. Visualize the Decision Tree
```c
from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15,8))
tree.plot_tree(
    model, filled=True, feature_names=X.columns, class_names=['No', 'Yes']
)
plt.title("Decision Tree Visualization (ID3 Algorithm)")
plt.show()
```
# Decision Tree Classifier using ID3 Algorithm

## Inputs

| Input | Description |
|-------|-------------|
| `decision_tree_dataset.csv` | Dataset containing features and target variable |
| Python Script | Implementation of ID3 Decision Tree |
| Optional Parameters | Tree depth, splitting criterion, etc. |

## Expected Outputs

| Output | Description |
|--------|-------------|
| Trained Model | Decision tree fitted to the dataset |
| Tree Visualization | Visual display of nodes and branches |
| Accuracy Score | Performance on unseen data |
| Confusion Matrix & Classification Report | Evaluation of predictions |

## Testing and Validation

- Test the model with known samples to ensure it predicts correctly.
- Use cross-validation to validate consistency.
- Tune tree depth using `max_depth` to avoid overfitting.
- Check feature importance with `model.feature_importances_`.

## Troubleshooting Tips

| Issue | Cause | Solution |
|-------|-------|---------|
| FileNotFoundError | Dataset path incorrect | Ensure the CSV is in the correct directory |
| ValueError: could not convert string to float | Non-numeric features | Encode categorical variables using `pd.get_dummies()` |
| Overfitting | Tree too deep | Set `max_depth` parameter |
| Low Accuracy | Poor feature correlation | Perform feature selection or improve data quality |

