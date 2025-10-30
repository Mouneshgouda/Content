# 🧠 Naïve Bayes Classifier Implementation on Sample Dataset

---

## 🎯 Objective
The goal of this project is to **implement a Naïve Bayes classifier** using Python and evaluate its performance on a given dataset.  
By training the model on labeled data stored in a `.csv` file and testing it with multiple test sets, this project demonstrates how probabilistic classification can be applied for prediction and accuracy measurement.

---

## 🎓 Learning Outcomes
After completing this project, you will be able to:
- Understand the working principle of the **Naïve Bayes algorithm** and its assumptions of feature independence.  
- Apply **data preprocessing**, encoding, and feature selection for classification tasks.  
- Build, train, and test a **Gaussian Naïve Bayes model** in Python.  
- Evaluate model performance using metrics such as **Accuracy**, **Precision**, **Recall**, and **F1-Score**.  
- Visualize confusion matrices and interpret classification performance.  
- Adapt the same methodology to solve **text, medical, or business classification problems**.

---

## 🧩 Basic Concepts Required
Before starting, ensure you understand the following concepts:
- **Probability Theory** — Conditional probability, Bayes’ theorem.  
- **Classification** — Difference between supervised learning and unsupervised learning.  
- **Feature Encoding** — Converting categorical data into numerical form.  
- **Model Evaluation Metrics** — Accuracy, precision, recall, and F1-score.  
- **Data Splitting** — Train-test split and cross-validation.  
- **Python Basics** — Reading CSV files, using pandas and sklearn.

---

## 🧰 Software and Libraries
You can execute this project using **Google Colab**, **Jupyter Notebook**, or any Python IDE.

### 🔧 Required Libraries:
Install or import the following libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
``

- **pandas** — for loading and preprocessing data  
- **numpy** — for numerical operations  
- **scikit-learn (sklearn)** — for model building and evaluation  
- **matplotlib / seaborn** — for visualization and accuracy comparison  

---

## 📂 Dataset

A sample CSV file (e.g., `training_data.csv`) containing both **features** and a **target class label**.

**Example structure:**
| Age | Salary | CreditScore | Purchased |
|-----|---------|--------------|------------|
| 25  | 40000   | 700          | Yes        |
| 35  | 60000   | 650          | No         |
| 29  | 52000   | 720          | Yes        |

---

## 🧩 Tasks and Step-by-Step Approach

### 🧠 Task 1: Load and Explore Dataset
**Goal:** Import dataset, understand its structure, and check for missing values.
```python
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load your dataset
df = pd.read_csv('training_data.csv')

# Create a LabelEncoder instance
le = LabelEncoder()

# Encode all object (string) columns automatically
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

print("All categorical columns encoded successfully!")
print(df.head())

https://drive.google.com/file/d/1Efhg2uFSIsc0mIHnBRgIQeubF1i8iIJ8/view?usp=drive_link
```
## 🧹 Task 2: Data Preprocessing
- Goal: Handle missing values, encode categorical data, and prepare features for training.
```python
df = pd.get_dummies(df, drop_first=True)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Separate features and target
X = df.drop('Purchased', axis=1)
y = df['Purchased']

https://drive.google.com/file/d/16D-TgIUUukkCdoIV1a1tVPsIO8elPBsa/view?usp=sharing
```
## 🧮 Task 3: Split Data into Train & Test Sets
- Goal: Separate data for training and testing the model.
```python
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

https://drive.google.com/file/d/1mfvdUT-bc4w3ed2BdSCOMhLvo8v9KgsS/view?usp=sharing
```
## 🤖 Task 4: Train the Naïve Bayes Classifier
- Goal: Train a Gaussian Naïve Bayes model on the training set.
```python
# Train Naïve Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

https://drive.google.com/file/d/1jGnBIVYaabcYsaZw4jxe0tdIM662xqJg/view?usp=sharing
```
## 🧾 Task 5: Test the Model
- Goal: Predict on test data and compute accuracy.
```python
# Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

https://drive.google.com/file/d/1oyUhFQXnJ71LagXA_rZOlVjkh6W2a23P/view?usp=sharing
```
## 📊 Task 6: Evaluate Model Performance
- Goal: Generate confusion matrix and classification report.
```python
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('📈 Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

https://drive.google.com/file/d/1aiTTWQ0NDpLB3FPg2SQ2JQ_wdQPuAwop/view?usp=sharing
```
## 🧾 Inputs

| Input | Description |
|--------|-------------|
| **training_data.csv** | Training dataset containing features and labels. |
| **test_data.csv (optional)** | Additional dataset for testing model performance. |
| **Python Script / Notebook** | Implementation of Naïve Bayes Classifier. |

---

## 🎯 Expected Outputs

| Output | Description |
|---------|-------------|
| **Trained Model** | Gaussian Naïve Bayes classifier fitted to training data. |
| **Accuracy Score** | Model performance on unseen data. |
| **Confusion Matrix & Report** | Evaluation metrics including precision, recall, and F1-score. |
| **Visualizations** | Accuracy heatmap and distribution plots. |

---

## 🔗 Google Colab Link

You can run this project online using Google Colab:  
👉 **[Open in Google Colab](https://colab.research.google.com/drive/1wBu1WyBI1GN6ngegPVnrwOqh0xRTjp5O?usp=sharing)**  


---

## 🧪 Testing and Validation

### ✅ 1. Test the Model with Known Samples
Use small test cases to ensure predictions align with known outcomes.

**Expected Behavior:**
- Model correctly predicts the majority of class labels.  
- Accuracy should typically range between **70%–95%**, depending on dataset quality.

---

### 📏 2. Validate Model Accuracy
Although Naïve Bayes is a simple model, you can still ensure robust performance by checking:

- **Data Balance:** Ensure both classes are evenly distributed.  
- **Cross-validation:** Use `cross_val_score` for performance stability.  
- **Feature Influence:** Observe which features contribute most to predictions.  

---

### 🧩 3. Troubleshooting Tips

| Issue | Likely Cause | Solution |
|--------|--------------|-----------|
| **FileNotFoundError** | Dataset not found in directory | Upload the CSV file to the correct path or Colab environment. |
| **ValueError** | Non-numeric values in feature columns | Encode categorical variables using `LabelEncoder` or `pd.get_dummies()`. |
| **Low Accuracy** | Insufficient data or noisy features | Remove outliers or try feature scaling. |
| **Empty Confusion Matrix** | Incorrect label encoding | Ensure the target column is encoded properly. |

---

💡 **Tip:**  
Always visualize feature distributions and correlations before training — Naïve Bayes assumes feature independence, which helps improve prediction accuracy.

## 💡 Similar Approach Can Be Used For Other Problems

---

### 🧠 Example 1: Email Spam Detection

**🎯 Objective:**  
Build a Naïve Bayes model to classify emails as **Spam** or **Not Spam** based on their content.

**📂 Dataset:**  
`emails.csv` — contains email text, sender info, and a label (`spam` / `ham`).

**🧩 Possible Tasks:**
1. Load and clean dataset (remove nulls and special characters).
2. Convert text to numerical features using **TF-IDF Vectorizer**.
3. Split into training and testing data.
4. Train a **Multinomial Naïve Bayes** model.
5. Evaluate accuracy, precision, and recall.
6. Visualize word frequency and confusion matrix.

**🎯 Expected Outputs:**
- Trained spam classifier.  
- Accuracy and classification report.  
- Insights on common spam words and sender patterns.

---

### 📊 Example 2: Medical Diagnosis – Disease Prediction

**🎯 Objective:**  
Use Naïve Bayes to predict whether a patient has a disease (e.g., diabetes) based on health indicators.

**📂 Dataset:**  
`medical_data.csv` — includes age, BMI, blood pressure, glucose level, and a `Disease` label (`Yes` / `No`).

**🧩 Possible Tasks:**
1. Load and clean the dataset (handle missing values).
2. Encode categorical features such as gender or smoking status.
3. Train a **Gaussian Naïve Bayes** model.
4. Predict disease presence for test data.
5. Evaluate using accuracy, confusion matrix, and ROC curve.
6. Visualize feature correlations with disease risk.

**🎯 Expected Outputs:**
- Disease prediction model with accuracy score.  
- Risk analysis charts.  
- Insights into key health features influencing outcomes.

---

### 🚀 You Can Also Try:
- Sentiment Analysis on Product Reviews  
- Credit Card Fraud Detection  
- Weather Forecast Classification (Sunny, Rainy, Cloudy)  
- Customer Churn Prediction  
- Loan Approval Classification  

---

These examples follow the **same Naïve Bayes workflow**:
> Load → Preprocess → Encode → Train → Evaluate → Visualize → Interpret

