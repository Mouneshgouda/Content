# ----------------------------------------------------------
# 🩺 HEART DISEASE DIAGNOSIS USING BAYESIAN NETWORK (FINAL SAFE VERSION)
# ----------------------------------------------------------
``` python
import pandas as pd
import numpy as np

# ✅ Fix for pgmpy + NumPy >= 2.0
if not hasattr(np, "product"):
    np.product = np.prod

from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# ----------------------------------------------------------
# 📂 Load Dataset
# ----------------------------------------------------------
df = pd.read_csv("heart_sample.csv")  # Replace with your CSV
print("✅ Dataset Loaded Successfully!\n")
print(df.head())

# ----------------------------------------------------------
# 🧹 Data Cleaning
# ----------------------------------------------------------
df.dropna(inplace=True)

# ----------------------------------------------------------
# 🧮 Discretize Continuous Variables
# ----------------------------------------------------------
df["age_cat"] = pd.cut(df["age"], bins=[0, 40, 55, 70, 100], labels=["Young", "Middle", "Old", "Elder"])
df["chol_cat"] = pd.cut(df["chol"], bins=[0, 200, 300, 600], labels=["Low", "Normal", "High"])
df["thalach_cat"] = pd.cut(df["thalach"], bins=[0, 120, 160, 220], labels=["Low", "Normal", "High"])

# Convert numeric discrete columns (like cp, sex, exang) into string categories
df["sex"] = df["sex"].astype(str)
df["cp"] = "cp_" + df["cp"].astype(str)
df["exang"] = df["exang"].astype(str)
df["target"] = df["target"].astype(str)

# Select only relevant columns
data = df[["age_cat", "sex", "cp", "chol_cat", "thalach_cat", "exang", "target"]].dropna()

print("\n✅ After Discretization and Conversion:")
print(data.head())

# Show available categories to confirm
print("\n🔎 Available Category Values:")
print("Age:", data["age_cat"].unique())
print("Cholesterol:", data["chol_cat"].unique())
print("Thalach:", data["thalach_cat"].unique())
print("CP:", data["cp"].unique())

# ----------------------------------------------------------
# 🧠 Define Bayesian Network
# ----------------------------------------------------------
model = BayesianModel([
    ("age_cat", "target"),
    ("sex", "target"),
    ("cp", "target"),
    ("chol_cat", "target"),
    ("thalach_cat", "target"),
    ("exang", "target")
])

# ----------------------------------------------------------
# ⚙️ Train Model
# ----------------------------------------------------------
model.fit(data, estimator=MaximumLikelihoodEstimator)
print("\n✅ Bayesian Network Model Trained Successfully!")

# ----------------------------------------------------------
# 🔍 Perform Inference (Safe & Compatible)
# ----------------------------------------------------------
inference = VariableElimination(model)

# Pick valid category values automatically
valid_age = str(data["age_cat"].unique()[0])
valid_chol = str(data["chol_cat"].unique()[0])
valid_thalach = str(data["thalach_cat"].unique()[0])
valid_cp = str(data["cp"].unique()[0])

print(f"\n🔍 Example Diagnosis Query:")
print(f"Using age={valid_age}, sex='1', cp={valid_cp}, chol={valid_chol}, thalach={valid_thalach}, exang='0'")

query_result = inference.query(
    variables=["target"],
    evidence={
        "age_cat": valid_age,
        "sex": "1",
        "cp": valid_cp,
        "chol_cat": valid_chol,
        "thalach_cat": valid_thalach,
        "exang": "0"
    }
)

print(query_result)

# ----------------------------------------------------------
# 💡 Interpretation
# ----------------------------------------------------------
print("\n💡 Interpretation:")
print("If P(target=1) > 0.5 → Patient likely has heart disease.")
print("If P(target=0) > 0.5 → Patient likely healthy.")
```
