# 🌳 Decision Tree Classifier using ID3 Algorithm

## 🎯 Objective
The objective of this project is to implement a **Decision Tree Classifier** using the **ID3 (Iterative Dichotomiser 3) algorithm** on a sample dataset.  
This project demonstrates how decision trees can be built using **entropy** and **information gain**, how to interpret the resulting model, and how to measure its accuracy on unseen data.

By analyzing categorical and numerical features, this model helps in making **interpretable, rule-based decisions** for real-world classification problems.

---

## 🎓 Learning Outcomes
After completing this project, you will be able to:
- Understand the concept of **entropy**, **information gain**, and **decision tree construction**.
- Build a decision tree using the **ID3 algorithm**.
- Visualize and interpret the decision-making process of a tree.
- Evaluate model accuracy using metrics like confusion matrix and classification report.
- Apply decision tree reasoning to other classification tasks.

---

## 🧠 Basic Concepts Required

| Concept | Description |
|----------|-------------|
| **Decision Tree** | A supervised learning model that makes decisions by splitting data using attributes. |
| **ID3 Algorithm** | A tree-building algorithm based on maximizing information gain at each split. |
| **Entropy** | A measure of impurity or randomness in data. |
| **Information Gain** | The reduction in entropy after a dataset is split on a particular attribute. |

---

## 📊 About the Dataset

**Dataset Name:** Decision Tree Sample Dataset  
**File:** `decision_tree_dataset.csv`  
**Records:** 200 samples  

| Feature | Description |
|----------|-------------|
| `Age_Group` | Categorical (Young / Middle-aged / Senior) |
| `Income_Level` | Categorical (Low / Medium / High) |
| `Owns_House` | Categorical (Yes / No) |
| `Marital_Status` | Categorical (Single / Married) |
| `Education` | Categorical (High School / Graduate / Post-Graduate) |
| `Purchased` | Target variable (1 = Purchased, 0 = Not Purchased) |

📌 *Dataset can be generated using the included Python script or replaced with your own CSV file.*

---

## ⚙️ Software and Libraries

| Library | Purpose |
|----------|----------|
| **Python 3.x** | Programming language |
| **pandas** | Data loading and manipulation |
| **numpy** | Numerical operations |
| **scikit-learn** | Decision Tree and metrics |
| **matplotlib** | Data visualization |
| **seaborn** | Enhanced plotting |

---

## 🧩 Tasks and Step-by-Step Approach

### 🧠 Task 1: Load and Explore Dataset
**Goal:** Understand dataset structure and features.

```python
import pandas as pd
df = pd.read_csv('decision_tree_dataset.csv')
print(df.head())
print(df.info())
print(df.describe())

https://drive.google.com/file/d/1Oc-rF66Uayv_6sv595doalOyXYJVyNsQ/view?usp=drive_link
```
## 🧹 Task 2: Data Preprocessing
- Goal: Handle missing values and encode categorical data.
```python
df.dropna(inplace=True)
df = pd.get_dummies(df, drop_first=True)
print(df.head())

https://drive.google.com/file/d/1jn7eqLgr_wj_5VYAKqEfKsGUD4KQFleT/view?usp=sharing
```
## 🧮 Task 3: Split Data into Features and Target
```python
X = df.drop('Purchased', axis=1)
y = df['Purchased']

https://drive.google.com/file/d/1qXEVTgfO3UI-yfT5goRNzqap9ohNkAlS/view?usp=sharing
```
## 🧩 Task 4: Split Dataset into Training and Testing Sets
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

https://drive.google.com/file/d/1uD0w1xK_k8RbNnpr_rbXEMYc0CuDTVUY/view?usp=sharing
```
## 🌳 Task 5: Train Decision Tree Model using ID3 Algorithm
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

https://drive.google.com/file/d/1PJhZJnUEt_RmW5dtQELf23v9qyuOZU3O/view?usp=sharing
```
## 🔍 Task 6: Evaluate Model
```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

https://drive.google.com/file/d/1mc1eF2vPwNmUOxQMMPtdOvePFh8G1JuU/view?usp=sharing
```

## 🌿 Task 7: Visualize the Decision Tree
```python
from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15,8))
tree.plot_tree(model, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.title("🌳 Decision Tree Visualization (ID3 Algorithm)")
plt.show()

https://drive.google.com/file/d/1o0OQ6b_PCF7eCfeiQ-1b302K5Tp-k21O/view?usp=sharing
```

## 🧾 Inputs

| Input | Description |
|--------|-------------|
| **decision_tree_dataset.csv** | Dataset containing categorical and numerical attributes |
| **Python Code** | Script implementing ID3 Decision Tree |
| **User Parameters (Optional)** | Tree depth, splitting criterion, etc. |

---

## 🎯 Expected Outputs

| Output | Description |
|---------|-------------|
| **Trained Model** | Decision Tree fitted to dataset |
| **Tree Visualization** | Visual display of branches and leaf nodes |
| **Accuracy Score** | Model performance on unseen data |
| **Confusion Matrix & Report** | Evaluation of true/false positives and negatives |

---

## 🔗 Google Colab Link

You can run this project online using Google Colab:  
👉 [**Open in Google Colab**](https://colab.research.google.com/drive/1K8mC4BpjVTlCST_XlbY0r7rTAShWJNmU?usp=sharing)  


---

## 🧪 Testing and Validation

### ✅ 1. Test the Model with Known Samples

**Expected Behavior:**
- Model correctly predicts the majority of outcomes.  
- Accuracy typically between **75%–95%**, depending on the dataset.

---

### 📏 2. Validate Model Accuracy

- **Cross-validation:** Use `cross_val_score()` to measure consistency.  
- **Tree depth tuning:** Set `max_depth` to avoid overfitting.  
- **Feature importance:** View using `model.feature_importances_`.  

---

### 🧩 3. Troubleshooting Tips

| Issue | Likely Cause | Solution |
|--------|---------------|-----------|
| **FileNotFoundError** | Dataset not in directory | Upload the dataset or correct file path. |
| **ValueError: could not convert string to float** | Non-numeric data not encoded | Use `pd.get_dummies()` before training. |
| **Overfitting** | Tree too deep | Limit depth using `max_depth`. |
| **Low Accuracy** | Poor feature correlation | Perform feature selection or improve data quality. |

## 💡 Similar Problems You Can Solve Using This Approach

### 📊 Example 1: Employee Attrition Prediction

**Objective:**  
Predict whether an employee is likely to leave a company based on various factors such as satisfaction, salary level, and work environment.

**Dataset:**  
`employee_attrition.csv` — contains columns like `Age`, `Department`, `Salary`, `YearsAtCompany`, `SatisfactionLevel`, and `Attrition` (Yes/No).

**Possible Tasks:**
- Load and clean the dataset (handle missing values).
- Encode categorical variables like Department and Salary.
- Split data into training and testing sets.
- Train a Decision Tree using **criterion='entropy'** (ID3 algorithm).
- Evaluate using accuracy, confusion matrix, and classification report.
- Visualize the decision tree.

**Expected Outputs:**
- Visual Decision Tree showing key factors leading to attrition.
- Accuracy report of how well the model predicts employee exits.
- Insights like:  
- “Employees with low satisfaction and long tenure are more likely to leave.”


---

### 🧠 Example 2: Loan Approval Prediction

**Objective:**  
Predict whether a loan application should be approved based on applicant details.

**Dataset:**  
`loan_approval.csv` — includes features like `ApplicantIncome`, `CreditScore`, `LoanAmount`, `Education`, `Married`, `Self_Employed`, and `Loan_Status`.

**Possible Tasks:**
- Load and preprocess dataset (handle nulls, encode categorical data).
- Split data into training and test sets.
- Train a Decision Tree using ID3 (entropy-based).
- Evaluate model accuracy and visualize decision logic.

**Expected Outputs:**
- Decision Tree visualization showing key splits like CreditScore and Income.
- Confusion Matrix showing model prediction results.
- Insights such as:  
- “Applicants with high credit scores and stable income have higher approval chances.”

