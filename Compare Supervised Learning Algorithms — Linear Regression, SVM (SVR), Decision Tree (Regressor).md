# Comparison of Supervised Learning Algorithms  
### (Linear Regression, Support Vector Machine, Decision Tree)

---

## Objective
The objective of this project is to compare three major supervised learning algorithms — Linear Regression, Support Vector Machine (SVM), and Decision Tree — on real-world data.  
We will evaluate their accuracy, interpretability, and predictive performance using a dataset containing employee demographic and job-related details.

---

## Learning Outcomes
After completing this project, you will be able to:
- Understand the fundamental differences between regression and classification models.  
- Implement Linear Regression, Support Vector Machine, and Decision Tree models in Python.  
- Perform preprocessing (encoding, scaling, and splitting) on real-world datasets.  
- Compare model performance using metrics like R² Score, Accuracy, Confusion Matrix**, and **Classification Report.  
- Visualize prediction results and interpret model behavior.

---

## Basic Concepts Required
Before you begin, ensure you are familiar with:
- Basics of Supervised Learning 
- Difference between Regression and Classification 
- Concepts of Overfitting, Accuracy, and Decision Boundaries  
- Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

## About the Dataset
You will use two datasets:

###  `supervised_regression_data.csv`
Predict Salary based on features:
| Feature | Description |
|----------|--------------|
| Age | Age of employee |
| Experience | Total years of experience |
| Education_Level | High School / Bachelor / Master |
| City | Location of employee |
| Salary | Annual income (target variable) |

###  supervised_classification_data.csv
Predict Attrition (Yes/No) based on similar attributes:
| Feature | Description |
|----------|--------------|
| Age | Age of employee |
| Experience | Total years of experience |
| Education_Level | Educational qualification |
| City | Location |
| Attrition | Whether employee left or not (target variable) |

---

## Software and Libraries

### Software Requirements
| Software | Description |
|-----------|-------------|
| Python (≥3.8)* | Core programming language used for model implementation |
| Google Colab / Jupyter Notebook | Interactive environment for running and visualizing results |
| Anaconda (optional) | Provides Python environment and package management |
| GitHub / Google Drive| For saving, sharing, and managing project files |

---

### Required Python Libraries
Install the required libraries before running the notebook:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn

```
## library Used
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score

https://drive.google.com/file/d/1SA1dZneaNNWUInHrnlJzeT5yA328mIJy/view?usp=sharing
```
## Tasks and Step-by-Step Approach
-  Task 1: Load and Explore Dataset
```python
df_reg = pd.read_csv('supervised_regression_data.csv')
df_cls = pd.read_csv('supervised_classification_data.csv')

print(df_reg.head())
print(df_cls.head())

https://drive.google.com/file/d/14GmW6yN-aPKwFIdmIIJ-fupfzT3BI5VA/view?usp=sharing
```
## Task 2: Data Preprocessing
- Convert categorical data into numeric using LabelEncoder and scale features.
```python
le = LabelEncoder()

for col in ['Education_Level', 'City']:
    df_reg[col] = le.fit_transform(df_reg[col])
    df_cls[col] = le.fit_transform(df_cls[col])

# Split features and targets
X_reg = df_reg[['Age', 'Experience', 'Education_Level', 'City']]
y_reg = df_reg['Salary']

X_cls = df_cls[['Age', 'Experience', 'Education_Level', 'City']]
y_cls = df_cls['Attrition']

# Encode classification target
y_cls = le.fit_transform(y_cls)

# Split datasets
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

https://drive.google.com/file/d/1JiPUudVxqF8lC9geC6pmDBvA0f27Ineo/view?usp=sharing
```
## Task 3: Linear Regression (Regression)
```python
reg = LinearRegression()
reg.fit(Xr_train, yr_train)
yr_pred = reg.predict(Xr_test)

print("R² Score (Linear Regression):", r2_score(yr_test, yr_pred))
sns.scatterplot(x=yr_test, y=yr_pred)
plt.title("Actual vs Predicted Salary (Linear Regression)")
plt.show()

https://drive.google.com/file/d/1ZiFr38uRzX7bAtsbfnIDOJOazErzezxE/view?usp=sharing
```
## Task 4: Support Vector Machine (Classification)
```python
svm = SVC(kernel='linear')
svm.fit(Xc_train, yc_train)
yc_pred_svm = svm.predict(Xc_test)

print("Accuracy (SVM):", accuracy_score(yc_test, yc_pred_svm))
print(confusion_matrix(yc_test, yc_pred_svm))
print(classification_report(yc_test, yc_pred_svm))

https://drive.google.com/file/d/1cc6ZZ1d1StSdZ0eOW2lv8M8Q4vM0h8KO/view?usp=sharing
```
## Task 5: Decision Tree (Classification)
```python
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(Xc_train, yc_train)
yc_pred_dt = dt.predict(Xc_test)

print("Accuracy (Decision Tree):", accuracy_score(yc_test, yc_pred_dt))
print(confusion_matrix(yc_test, yc_pred_dt))
print(classification_report(yc_test, yc_pred_dt))

https://drive.google.com/file/d/1jpgKn6MEAjYPJIUr3Q1ghGzgn75OkCs4/view?usp=sharing
```
## Task 6: Compare Results

| Model | Type | Metric | Score |
|--------|------|---------|-------|
| Linear Regression | Regression | R² Score | ~0.80–0.90 |
| SVM | Classification | Accuracy | ~0.80–0.95 |
| Decision Tree | Classification | Accuracy | ~0.75–0.90 |

---

## Inputs

| Input | Description |
|--------|-------------|
| `supervised_regression_data.csv` | Dataset for Salary prediction |
| `supervised_classification_data.csv` | Dataset for Attrition prediction |
| Parameters | Split ratio, random seed, depth, kernel type |

---

## Expected Outputs

| Output | Description |
|---------|-------------|
| Regression Plot | Salary prediction vs actual |
| Classification Reports | Accuracy, Precision, Recall, F1-Score |
| Confusion Matrices | SVM & Decision Tree results |
| Comparison Table | Summary of all model performances |

---

## Google Colab Link

You can run this project online using Google Colab:  
[**Open in Google Colab**](https://colab.research.google.com/drive/1pbatbx4FQFrttHBYjm_3BspHFJzwMrZ5?usp=sharing)  

---

## Testing and Validation

### 1. Test with Known Patterns
- Salary increases with Experience.  
- Employees with **higher education** tend to earn more.  
- Attrition may be higher among younger, less-experienced employees.

### 2. Validate Model Accuracy
- Use `cross_val_score()` for model stability.  
- Tune hyperparameters (e.g., `max_depth`, `kernel`, `C`) for better results.  

---

## Troubleshooting Tips

| Issue | Likely Cause | Solution |
|--------|---------------|-----------|
| `FileNotFoundError` | File not in directory | Verify dataset path or upload to Colab |
| `ValueError: could not convert string to float` | Unencoded categorical data | Use `LabelEncoder()` or `get_dummies()` |
| Low Accuracy | Overfitting or unscaled data | Adjust `max_depth` or apply scaling |
| Empty predictions | Wrong target selection | Ensure correct target column is used |

