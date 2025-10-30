# 🎇 Data Analysis & Visualization on Diwali Sales Dataset

## 🎯 Objective

The objective of this project is to perform **data analysis and visualization** on the **Diwali Sales Dataset** using Python.  
By analyzing customer demographics, purchase behavior, and product categories, the project aims to extract **business insights** that can help improve marketing strategies, target specific customer groups, and boost overall sales during festive seasons.

---

## 🎓 Learning Outcomes

After completing this project, you will be able to:

1. 📊 Understand and apply **data analysis techniques** using Python libraries such as Pandas and NumPy.  
2. 🧹 Perform **data cleaning and preprocessing**, including handling missing values and data type conversions.  
3. 📈 Visualize data using **Matplotlib** and **Seaborn** to identify sales trends and customer patterns.  
4. 🧠 Derive **insights** from data to support business decision-making.  
5. 💡 Communicate findings effectively using charts and summaries.

---

## 📘 Basic Concepts Required

| Concept | Description |
|----------|-------------|
| **Data Cleaning** | Removing or correcting inaccurate, missing, or irrelevant data to ensure quality results. |
| **Data Visualization** | Representing data through visual elements like charts and graphs to easily identify patterns and trends. |
| **Exploratory Data Analysis (EDA)** | Summarizing the main characteristics of a dataset using statistical and visual methods. |
| **Grouping & Aggregation** | Summarizing data by categories (e.g., total sales by state or gender) using `groupby()` in Pandas. |
| **Correlation** | A statistical measure showing how two variables relate to each other (e.g., between Orders and Amount). |
| **Customer Segmentation** | Dividing customers into groups based on demographic or behavioral attributes for better targeting. |

---

## 🧾 About the Dataset

**Dataset Name:** Diwali Sales Data  
**Source:** Publicly available on [Kaggle](https://www.kaggle.com/) or provided as a `.csv` file.  

### 📂 Dataset Description:

The dataset contains sales transactions from an e-commerce platform during the **Diwali Festival** period.  
It includes customer demographics, purchase details, and product categories.

| Column Name | Description |
|--------------|-------------|
| `User_ID` | Unique identifier for each customer |
| `Cust_name` | Customer name |
| `Gender` | Gender of the customer (Male/Female) |
| `Age Group` | Age category of the customer |
| `Age` | Actual age in years |
| `Marital_Status` | Marital status (Married/Single) |
| `State` | State from which the customer made a purchase |
| `Occupation` | Profession of the customer |
| `Product_Category` | Category of the purchased product |
| `Orders` | Number of orders placed |
| `Amount` | Total purchase amount |

**Goal:** 

- To analyze purchasing trends across demographics and product categories during the festive season.


## 🧰 Software and Libraries Used

| Tool / Library | Purpose |
|-----------------|----------|
| **Python** | Programming language for data analysis and visualization |
| **Pandas** | Data manipulation and cleaning |
| **NumPy** | Numerical computations and array operations |
| **Matplotlib** | Basic data visualization |
| **Seaborn** | Advanced and aesthetic statistical visualization |
| **Jupyter Notebook / Google Colab** | Interactive environment for executing Python code |

---

## 🧩 Tasks and Step-by-Step Approach

### 🧠 Task 1: Load and Explore Dataset
**Goal:** Understand the dataset structure and identify key attributes.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('Diwali Sales Data.csv', encoding='unicode_escape')
print(df.head())
print(df.info())
print(df.describe())

https://drive.google.com/file/d/1Vj5zrXCaZyDrMKQDPKdy8gT4MftqhM1z/view?usp=drive_link
```

## 🧹 Task 2: Data Cleaning
- Goal: Remove unnecessary columns, handle missing values, and fix data types.

```python
df.drop(columns=['Status', 'unnamed1'], errors='ignore', inplace=True)
df.dropna(inplace=True)
df['Amount'] = df['Amount'].astype('int')

https://drive.google.com/file/d/1GZk9CgF4peEd_Vi_Mc9FOQUEitB97wML/view?usp=drive_link
```
## 👥 Task 3: Age Group Analysis
- Goal: Analyze total sales across age groups and gender.
```python
  sns.barplot(x='Age Group', y='Amount', hue='Gender', data=df, estimator=sum)
plt.title('👥 Total Sales by Age Group and Gender')
plt.ylabel('Total Amount')
plt.show()

https://drive.google.com/file/d/17AyRL_q5I5cQKLr8HZJpwwqVsuKpgdm0/view?usp=drive_link
```
## 💼 Task 4: Occupation-Based Sales
- Goal: Analyze how profession influences total purchases.
```python
sns.barplot(x='Occupation', y='Amount', data=df, estimator=sum)
plt.title('💼 Total Sales by Occupation')
plt.xticks(rotation=45)
plt.ylabel('Total Amount')
plt.show()

https://drive.google.com/file/d/1GoWexaXPIzfzxYnW5f-9ddL9K0kBQiti/view?usp=drive_link
```
## 🛍️ Task 5: Product Category Analysis
- Goal: Identify top-performing product categories.
```python
sns.barplot(x='Product_Category', y='Amount', data=df, estimator=sum)
plt.title('🛍️ Total Sales by Product Category')
plt.xticks(rotation=45)
plt.ylabel('Total Amount')
plt.show()

https://drive.google.com/file/d/1d0sbrZkl5WhXF0Lp1GqMKWBk7hKgDhFO/view?usp=drive_link
```
## 📦 Task 6: Top 10 Most Sold Products
- Goal: Find most frequently purchased products.
```python
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues')
plt.title('📈 Correlation Heatmap')
plt.show()

https://drive.google.com/file/d/1vRAn7pcfT0RnQ1OBVWyUsTNd5vDPtA9I/view?usp=drive_link
```
## 📈 Task 7: Correlation Heatmap
- Goal: Examine relationships among numerical variables.
```python
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues')
plt.title('📈 Correlation Heatmap')
plt.show()

https://drive.google.com/file/d/15AiQ0f2-CpMoyCPPPeMgPwssdbia-0oI/view?usp=drive_link
```
## 🧾 Inputs

| **Input** | **Description** |
|------------|----------------|
| **Diwali Sales Data.csv** | Dataset containing customer demographics, purchase details, and sales amounts. |
| **Python Code** | Script or notebook implementing data analysis and visualization. |
| **User Parameters (Optional)** | Filters for Age Group, Gender, State, etc., to customize analysis. |

---

## 🎯 Expected Outputs

| **Output** | **Description** |
|-------------|----------------|
| **Cleaned Dataset** | Preprocessed data ready for visualization. |
| **Visualizations** | Charts and plots showing trends and patterns. |
| **Insights Summary** | Business insights derived from data (age group, top states, etc.). |
| **Correlation Matrix** | Relationships between numerical columns. |

---

## 🔗 Google Colab Link

You can run this project online using **Google Colab**:  
👉 [**Open in Google Colab**](https://colab.research.google.com/drive/1lMCfn0Kq8DypMGhXrb2B58h3otnsh7Bm?usp=sharing)


---

## 🧪 Testing and Validation

### ✅ 1. Test the Model with Known Samples

Run the analysis with known patterns to verify realistic outcomes.

**Expected Behavior:**
- Female customers have higher purchase totals.  
- The **26–35 age group** contributes the most to sales.  
- **Maharashtra, Karnataka, and Uttar Pradesh** lead in total sales.

---

### 📏 2. Validate the Model Accuracy (Optional)

Although this is **not a machine learning model**, you can check the accuracy and reliability of your results through:

- **Data Completeness:** Ensure no missing values remain.  
- **Consistency:** Charts reflect expected sales behavior.  
- **Cross-Verification:** Compare results with pivot tables or SQL queries for accuracy.

---

### 🧩 3. Troubleshooting Tips

| **Issue** | **Likely Cause** | **Solution** |
|------------|------------------|--------------|
| `FileNotFoundError` | Dataset not in working directory | Verify the dataset path or upload to Colab. |
| `KeyError: column not found` | Incorrect column name | Check column headers using `df.columns`. |
| `ValueError: cannot convert` | Non-numeric data in `Amount` column | Clean and convert data type using `astype(int)`. |
| **Empty Graphs** | Incorrect `groupby` or aggregation | Ensure correct columns are passed to `groupby()` and plotting functions. |

---

## 💡 Similar Problems You Can Solve Using This Approach

The step-by-step data analysis and visualization process used in the **Diwali Sales Dataset** project can be applied to other datasets as well.  
Below are two example projects that follow a similar workflow — from data cleaning and exploration to visualization and insights generation.

---

### 🛍️ Example 1: E-commerce Customer Behavior Analysis

#### 🎯 Objective
Analyze customer purchase behavior from an online shopping dataset to understand what drives repeat purchases and identify factors influencing total spending.

#### 📂 Dataset
**Ecommerce_Customers.csv** — includes:
- Customer ID  
- Session length  
- Time on App  
- Time on Website  
- Pages viewed  
- Amount spent  

#### 🧩 Possible Tasks
1. **Load and clean data** — handle missing values and correct data types.  
2. **Explore correlations** between time spent, session length, and purchase amount.  
3. **Visualize user engagement** (App vs. Website) and its impact on total spending.  
4. **Identify high-value customer segments** based on behavior patterns.  
5. **Generate business insights** such as which platform contributes more to revenue.  

#### 🎯 Expected Output
- 📊 Visuals showing whether the App or Website drives higher revenue.  
- 📈 Correlation heatmaps between engagement metrics and spending.  
- 💡 Insights into which customer behaviors lead to increased purchases.  

---

### 🎓 Example 2: Student Performance Analysis

#### 🎯 Objective
Study the effect of gender, study hours, and parental education on student exam performance using Python-based data analysis.

#### 📂 Dataset
**StudentsPerformance.csv** — includes:
- Gender  
- Parental education level  
- Lunch type  
- Test preparation course  
- Math, reading, and writing scores  

#### 🧩 Possible Tasks
1. **Load and explore dataset** to understand structure and attributes.  
2. **Clean data** by removing missing or inconsistent values.  
3. **Analyze average scores** by gender, study habits, and parental education.  
4. **Generate correlation heatmap** for math, reading, and writing scores.  
5. **Visualize performance distribution** across different demographic groups.  

#### 🎯 Expected Output
- 📊 Charts comparing performance by gender and parental background.  
- 📈 Correlation between subject scores to understand learning patterns.  
- 💡 Insights on how study preparation and family background affect student success.

---

📘 **Note:**  
You can replicate this workflow in Google Colab or Jupyter Notebook using libraries like **pandas**, **numpy**, **matplotlib**, and **seaborn** — just like in the Diwali Sales project.  
Simply replace the dataset and modify the column names as per your data structure.


