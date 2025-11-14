# Artificial Neural Network using Backpropagation  

---

## Objective
To build and train an **Artificial Neural Network (ANN)** using the **Backpropagation Algorithm** to predict **customer churn** (whether a customer will leave the bank or not).  
We use the **Churn Modelling dataset** from Kaggle to train, test, and evaluate model performance.

---

## Learning Outcomes
- Understand how a basic **ANN** works using **Backpropagation**.  
- Perform **data preprocessing** for real-world datasets.  
- Evaluate model performance using metrics such as **accuracy** and **loss**.  
- Visualize learning curves and make predictions using Keras.

---

## Basic Concepts Required
- Neural network structure: Input, Hidden, and Output layers  
- Activation functions: ReLU, Sigmoid  
- Gradient Descent and Backpropagation  
- Evaluation metrics: Accuracy, Confusion Matrix, Loss Curve  

---

## About the Dataset
**Dataset Name:** `Churn_Modelling.csv`  
**Source:** Kaggle  

### Description  
This dataset contains customer demographics and banking details to predict whether a customer will leave the bank (churn).  

| Column | Description |
|--------|--------------|
| CustomerId | Unique customer ID |
| Surname | Customer surname |
| CreditScore | Credit score of the customer |
| Geography | Country (France, Spain, Germany) |
| Gender | Male/Female |
| Age | Customer age |
| Tenure | Number of years as a bank member |
| Balance | Account balance |
| NumOfProducts | Number of bank products |
| HasCrCard | 1 if customer has a credit card, else 0 |
| IsActiveMember | 1 if active, else 0 |
| EstimatedSalary | Customer’s estimated salary |
| Exited | Target — 1 if customer left, else 0 |

---

## Software and Libraries

### Software Requirements

| Software | Description |
|-----------|-------------|
| **Python (≥3.8)** | Primary programming language used for model implementation |
| **Google Colab / Jupyter Notebook** | Interactive IDE for coding, visualization, and training neural networks |
| **Anaconda (optional)** | For environment management and package installation |
| **GitHub / Google Drive** | Store and share datasets, scripts, and project notebooks |
| **Kaggle** | Source for the `Churn_Modelling.csv` dataset |

---

### Required Python Libraries

Install all dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras

```
## Tasks and Step-by-Step Approach
-  Task 1: Load and Explore Dataset
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('Churn_Modelling.csv')
print(df.head())
print(df.info())
print(df.describe())

https://drive.google.com/file/d/1eDjwVw8p4PKsKg0Kn5lIpoWZi8HzrLVQ/view?usp=sharing
```
## 🧹 Task 2: Data Preprocessing
- Drop unnecessary columns (RowNumber, CustomerId, Surname)
- Encode categorical data (Gender, Geography)
- Split into features (X) and labels (y)
- Normalize numeric data for stable ANN training
```python
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Drop irrelevant columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encode categorical features
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

# Split features and labels
X = df.drop('Exited', axis=1).values
y = df['Exited'].values

# Feature scaling
sc = StandardScaler()
X = sc.fit_transform(X)

https://drive.google.com/file/d/1ewRID0qNciXtGuepBnaergGjg17bauVm/view?usp=sharing
```
## Task 3: Build the ANN Model
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Initialize the ANN
model = Sequential()

# Input + Hidden Layers
model.add(Dense(units=6, activation='relu', input_dim=X.shape[1]))
model.add(Dense(units=6, activation='relu'))

# Output Layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

https://drive.google.com/file/d/1wuci6cZ4gqb-1mOYbWfGFZr3bQj0N1M4/view?usp=sharing
```
## 🚀 Task 4: Train the Model
```python
history = model.fit(X, y, batch_size=32, epochs=50, validation_split=0.2)

https://drive.google.com/file/d/1r1ZEObl3UnW8BSQaQy5eywCqu04ydFNF/view?usp=sharing
```
##  Task 5: Visualize Accuracy and Loss
```python
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.legend()
plt.show()

https://drive.google.com/file/d/1OG1k6IxfQY997wgJt3M-kv4cts1O8rLa/view?usp=sharing
```
##  Task 6: Evaluate the Model
```python
from sklearn.metrics import confusion_matrix, classification_report
y_pred = (model.predict(X) > 0.5).astype(int)
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))

https://drive.google.com/file/d/1L0cQN1Zf7pJUqEozKe4VTMHDMYavTQCP/view?usp=sharing
```
##  Inputs

| Input | Description |
|--------|-------------|
| `Churn_Modelling.csv` | Dataset containing customer information |
| Python Script / Notebook | Implementation of ANN using Keras |
| Parameters | Batch size, epochs, activation functions, optimizer |

---

##  Expected Outputs

| Output | Description |
|--------|-------------|
| Trained ANN | Model fitted using backpropagation |
| Accuracy Score | Model performance (typically ~80–85%) |
| Loss Curves | Training and validation loss visualization |
| Confusion Matrix | Evaluation of prediction accuracy |

---

##  Google Colab Link

Run this project online using Google Colab:  
👉 [**Open in Google Colab**](https://colab.research.google.com/drive/1Vxeja1UEweJElQkiNCR34b44rDoUMCpw?usp=sharing)  

---

##  Testing and Validation

###  1. Test with Known Patterns
- Customers with **low balance** or **low activity** are more likely to churn.  
- **Higher credit score** or **longer tenure** → lower churn rate.  

###  2. Validate Model Accuracy
- Use `model.evaluate(X_test, y_test)` if dataset is split.  
- Adjust **hidden layers**, **neurons**, and **epochs** to optimize performance.  

---

##  Troubleshooting Tips

| Issue | Likely Cause | Solution |
|--------|---------------|-----------|
| `ValueError: could not convert string to float` | Unencoded categorical data | Use `LabelEncoder` or `get_dummies()` |
| Low Accuracy | Underfitting or insufficient layers | Increase hidden units or epochs |
| Overfitting | Too many epochs / no dropout | Add dropout or early stopping |
| Slow Training | Large dataset or high epochs | Reduce epochs or use GPU in Colab |

##  Example 1: Bank Customer Churn Prediction

**Objective:**  
Predict whether a customer will leave the bank (churn) based on attributes such as credit score, age, balance, and activity level.

**Dataset:**  
[`Churn_Modelling.csv`](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) — includes customer demographics and banking details.

**Possible Tasks:**
- Load and preprocess the dataset.  
- Encode categorical variables (e.g., Geography, Gender).  
- Normalize numerical features.  
- Build an **Artificial Neural Network** using **Keras** with 2–3 hidden layers.  
- Train using **backpropagation** and **binary cross-entropy loss**.  
- Evaluate accuracy, loss, and confusion matrix.  

**Expected Output:**
- Accuracy around **80–85%**.  
- Loss curve showing model convergence.  
- Confusion matrix identifying true vs. predicted churners.  
- Insight: Customers with lower credit scores or balances are more likely to churn.

---

##  Example 2: Diabetes Prediction using ANN

**Objective:**  
Build an ANN to classify whether a patient has diabetes based on medical attributes.

**Dataset:**  
[`Pima Indians Diabetes Database`](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

**Possible Tasks:**
- Load and clean the dataset.  
- Normalize glucose, BMI, and age values.  
- Split data into training and test sets.  
- Construct an ANN with ReLU activations and Sigmoid output.  
- Train the model with **Adam optimizer** and **binary cross-entropy** loss.  
- Evaluate performance on test data.  

**Expected Output:**
- Accuracy ~78–85%.  
- Visualizations of loss and accuracy per epoch.  
- Insight: Higher glucose levels and BMI correlate strongly with diabetes.  

---


