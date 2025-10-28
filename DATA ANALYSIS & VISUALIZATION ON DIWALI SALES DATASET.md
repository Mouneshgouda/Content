# 🎇 Data Analysis & Visualization on Diwali Sales Dataset

This project performs **data analysis and visualization** on the *Diwali Sales Dataset* using Python libraries like **Pandas**, **NumPy**, **Matplotlib**, and **Seaborn**.

---

## 📦 Libraries Used
- **pandas** — for data loading and manipulation  
- **numpy** — for numerical operations  
- **matplotlib** — for data visualization  
- **seaborn** — for advanced, beautiful graphs  

---

## 📂 Load and Explore Dataset
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)

# Load the dataset
df = pd.read_csv('Diwali Sales Data.csv', encoding='unicode_escape')
print("✅ Dataset Loaded Successfully!\n")
print(df.head())
```

## 🧹 Data Cleaning
```python
# Drop unnecessary columns (if present)
df.drop(columns=['Status', 'unnamed1'], errors='ignore', inplace=True)

# Drop missing values
df.dropna(inplace=True)

# Convert Amount column to integer
df['Amount'] = df['Amount'].astype('int')

# Display dataset info
print("\n📋 Dataset Info:")
print(df.info())
```
## 👥 2. Age Group Analysis
```python
sns.barplot(x='Age Group', y='Amount', hue='Gender', data=df, estimator=sum)
plt.title('👥 Total Sales by Age Group and Gender')
plt.ylabel('Total Amount')
plt.show()
```
## 🏙️ 3. State-Wise Sales
```python
sales_state = df.groupby('State')['Amount'].sum().sort_values(ascending=False).head(10)
print(sales_state)

sns.barplot(x=sales_state.index, y=sales_state.values)
plt.title('🏙️ Top 10 States by Total Sales')
plt.xticks(rotation=45)
plt.ylabel('Total Sales Amount')
plt.show()
```
## 💼 4. Occupation-Based Sales
```python
sales_occupation = df.groupby('Occupation')['Amount'].sum().sort_values(ascending=False)
print(sales_occupation)

sns.barplot(x='Occupation', y='Amount', data=df, estimator=sum)
plt.title('💼 Total Sales by Occupation')
plt.xticks(rotation=45)
plt.ylabel('Total Amount')
plt.show()

```
## 🛍️ 5. Product Category Analysis
```python
sales_category = df.groupby('Product_Category')['Amount'].sum().sort_values(ascending=False)
print(sales_category)

sns.barplot(x='Product_Category', y='Amount', data=df, estimator=sum)
plt.title('🛍️ Total Sales by Product Category')
plt.xticks(rotation=45)
plt.ylabel('Total Amount')
plt.show()
```
## 📦 6. Top 10 Most Sold Products
``` python
top_products = df.groupby('Product_ID')['Orders'].sum().sort_values(ascending=False).head(10)
print(top_products)

sns.barplot(x=top_products.index, y=top_products.values)
plt.title('📦 Top 10 Most Sold Products')
plt.xlabel('Product ID')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45)
plt.show()
```
## 📈 7. Correlation Heatmap (Numerical Columns)
```python
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues')
plt.title('📈 Correlation Heatmap')
plt.show()
```
## 💡 Insights Summary

1️⃣ **Female customers** tend to purchase more than **male customers**.  

2️⃣ Most buyers are from the **26–35 age group**.  

3️⃣ **Maharashtra**, **Karnataka**, and **Uttar Pradesh** show the **highest sales**.  

4️⃣ Top professions contributing to sales: **IT**, **Healthcare**, and **Aviation**.  

5️⃣ Most popular product categories: **Food**, **Clothing**, and **Electronics**.  

6️⃣ There is a **strong positive correlation** between **Orders** and **Amount**.
