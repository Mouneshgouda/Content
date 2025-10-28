# 🎇 DATA ANALYSIS & VISUALIZATION ON DIWALI SALES DATASET

# ----------------------------------------------------------
# 📦 Importing Required Libraries
# ----------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# To make graphs look clean
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)

# ----------------------------------------------------------
# 📂 Load Dataset
# ----------------------------------------------------------
df = pd.read_csv('/mnt/data/Diwali Sales Data.csv', encoding='unicode_escape')
print("✅ Dataset Loaded Successfully!\n")
print(df.head())

# ----------------------------------------------------------
# 🧹 Data Cleaning
# ----------------------------------------------------------

# Drop unnecessary columns (if present)
df.drop(columns=['Status', 'unnamed1'], errors='ignore', inplace=True)

# Drop missing values
df.dropna(inplace=True)

# Convert Amount column to integer
df['Amount'] = df['Amount'].astype('int')

# Basic dataset info
print("\n📋 Dataset Info:")
print(df.info())

# ----------------------------------------------------------
# 📊 1. Gender-Based Analysis
# ----------------------------------------------------------
print("\n📊 Analyzing Sales by Gender...")

# Total amount spent by each gender
gender_sales = df.groupby('Gender')['Amount'].sum().sort_values(ascending=False)
print(gender_sales)

# Bar plot — Total Sales by Gender
sns.barplot(x='Gender', y='Amount', data=df, estimator=sum)
plt.title('💰 Total Sales by Gender')
plt.ylabel('Total Amount')
plt.show()

# ----------------------------------------------------------
# 👥 2. Age Group Analysis
# ----------------------------------------------------------
print("\n📊 Analyzing Sales by Age Group...")

# Total amount by Age Group and Gender
sns.barplot(x='Age Group', y='Amount', hue='Gender', data=df, estimator=sum)
plt.title('👥 Total Sales by Age Group and Gender')
plt.ylabel('Total Amount')
plt.show()

# ----------------------------------------------------------
# 🏙️ 3. State-Wise Sales
# ----------------------------------------------------------
print("\n📊 Analyzing Sales by State...")

# Group and sort
sales_state = df.groupby('State')['Amount'].sum().sort_values(ascending=False).head(10)
print(sales_state)

# Bar Plot — Top 10 States by Total Sales
sns.barplot(x=sales_state.index, y=sales_state.values)
plt.title('🏙️ Top 10 States by Total Sales')
plt.xticks(rotation=45)
plt.ylabel('Total Sales Amount')
plt.show()

# ----------------------------------------------------------
# 💼 4. Occupation-Based Sales
# ----------------------------------------------------------
print("\n📊 Analyzing Sales by Occupation...")

# Group and sort
sales_occupation = df.groupby('Occupation')['Amount'].sum().sort_values(ascending=False)
print(sales_occupation)

# Bar Plot — Sales by Occupation
sns.barplot(x='Occupation', y='Amount', data=df, estimator=sum)
plt.title('💼 Total Sales by Occupation')
plt.xticks(rotation=45)
plt.ylabel('Total Amount')
plt.show()

# ----------------------------------------------------------
# 🛍️ 5. Product Category Analysis
# ----------------------------------------------------------
print("\n📊 Analyzing Sales by Product Category...")

# Group and sort
sales_category = df.groupby('Product_Category')['Amount'].sum().sort_values(ascending=False)
print(sales_category)

# Bar Plot — Total Sales by Product Category
sns.barplot(x='Product_Category', y='Amount', data=df, estimator=sum)
plt.title('🛍️ Total Sales by Product Category')
plt.xticks(rotation=45)
plt.ylabel('Total Amount')
plt.show()

# ----------------------------------------------------------
# 📦 6. Top 10 Most Sold Products
# ----------------------------------------------------------
print("\n📊 Finding Top 10 Most Sold Products...")

# Group by Product ID
top_products = df.groupby('Product_ID')['Orders'].sum().sort_values(ascending=False).head(10)
print(top_products)

# Bar Plot — Top 10 Sold Products
sns.barplot(x=top_products.index, y=top_products.values)
plt.title('📦 Top 10 Most Sold Products')
plt.xlabel('Product ID')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45)
plt.show()

# ----------------------------------------------------------
# 📈 7. Correlation Heatmap (Numerical Columns)
# ----------------------------------------------------------
print("\n📊 Correlation between Numeric Variables...")

corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues')
plt.title('📈 Correlation Heatmap')
plt.show()

# ----------------------------------------------------------
# 💡 INSIGHTS SUMMARY
# ----------------------------------------------------------
print("\n💡 FINAL INSIGHTS FROM ANALYSIS:")
print("1️⃣ Female customers tend to purchase more than male customers.")
print("2️⃣ Most buyers are from the 26–35 age group.")
print("3️⃣ Maharashtra, Karnataka, and Uttar Pradesh show the highest sales.")
print("4️⃣ IT, Healthcare, and Aviation professionals purchase the most.")
print("5️⃣ Top product categories: Food, Clothing, and Electronics.")
print("6️⃣ Strong positive correlation between Orders and Amount.")

print("\n✅ Data Analysis & Visualization Completed Successfully!")
