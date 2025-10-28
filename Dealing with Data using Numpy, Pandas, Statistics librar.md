# 📘 Dealing with Data using NumPy, Pandas, and Statistics Library

This guide covers how to handle and analyze data in Python using three essential libraries — **NumPy**, **Pandas**, and **Statistics**.  
These tools are the foundation of data analysis, data science, and machine learning workflows.

---

## 🧠 Overview

When working with datasets in Python, these libraries are used together for efficient, reliable, and fast data processing:

| Library | Purpose | Key Strengths |
|----------|----------|----------------|
| **NumPy** | Numerical computations | Fast array operations, vectorization |
| **Pandas** | Data manipulation & analysis | Works with structured/tabular data |
| **Statistics / SciPy** | Statistical analysis | Summaries, hypothesis tests, and correlations |

---

## 🧩 1. NumPy — Numerical Python

### 🔍 What is NumPy?
**NumPy** provides fast and efficient array operations, replacing slow Python loops with optimized vectorized computations.

### ⚙️ Installation
```bash
pip install numpy
```

### Example 

```python
import numpy as np

# Create arrays
a = np.array([1, 2, 3, 4, 5])
b = np.arange(10)          # 0 to 9
c = np.random.rand(3, 3)   # 3x3 matrix of random numbers

# Array operations
print(a + 10)              # Add 10 to every element
print(a ** 2)              # Square each element

# Statistical functions
print(np.mean(a))          # Mean
print(np.std(a))           # Standard deviation
print(np.sum(a))           # Sum
```
### ✅ Why Use NumPy?

- Performs fast, vectorized computations  
- Foundation for Pandas and most data science libraries  
- Efficient memory usage and mathematical operations  

## 📊 2. Pandas — Data Analysis Library

### 🔍 What is Pandas?

**Pandas** makes it easy to work with structured (tabular) data, similar to Excel spreadsheets.  
It provides **Series** (1D) and **DataFrame** (2D) objects.

---

### ⚙️ Installation
```bash
pip install pandas
```
### Example
```python
import pandas as pd

# Create DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'Salary': [50000, 60000, 70000, 80000]
}
df = pd.DataFrame(data)

# View data
print(df.head())       # Display first few rows
print(df.describe())   # Summary statistics

# Selecting data
print(df['Name'])                  # Single column
print(df[df['Age'] > 30])          # Filter rows

# Add a new column
df['Tax'] = df['Salary'] * 0.1

# Grouping and aggregation
grouped = df.groupby('Age').mean()
print(grouped)
```
## Reading and Writing Files
```python
# Save to CSV
df.to_csv('output.csv', index=False)

# Load CSV
df2 = pd.read_csv('output.csv')
```
### ✅ Why Use Pandas?

- Simplifies data cleaning, filtering, grouping, and joining  
- Easily handles large datasets  
- Integrates well with NumPy, Matplotlib, and other tools  

## 📈 3. Statistics — Descriptive & Inferential Analysis

### 🔍 What is the Statistics Library?

Python’s built-in **statistics** module and **SciPy’s stats** module help perform mathematical and statistical computations such as mean, median, mode, variance, and hypothesis testing.

---

### ⚙️ Installation (for SciPy)
```bash
pip install scipy
```
## Example Usage
- Using Built-in statistics:
```python
import statistics as stats

data = [10, 20, 30, 40, 50]

print(stats.mean(data))     # Mean
print(stats.median(data))   # Median
print(stats.mode(data))     # Mode
print(stats.stdev(data))    # Standard deviation
```
- Using scipy.stats:
  ```python
  from scipy import stats
import numpy as np

data = [1, 2, 2, 3, 4, 4, 4, 5]

# Descriptive statistics
print(np.mean(data))
print(stats.mode(data, keepdims=True))
print(np.var(data))

# Hypothesis testing (One-sample t-test)
t_stat, p_val = stats.ttest_1samp(data, 3)
print(f"T-statistic: {t_stat}, p-value: {p_val}")
```
### ✅ Why Use Statistics Libraries?

- Summarize data effectively  
- Analyze distributions and variance  
- Conduct hypothesis testing (e.g., t-test, chi-square, ANOVA)  


