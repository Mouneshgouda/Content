## 📰 Naïve Bayes Classifier for Text Document Classification

### 🔍 What is Naïve Bayes for Text Classification?

**Naïve Bayes** is a probabilistic algorithm based on **Bayes’ Theorem**, widely used for **document and email classification**, **spam filtering**, and **sentiment analysis**.  
It calculates the probability of a document belonging to a certain class based on the frequency of words in the text.  

The variant we’ll use is **Multinomial Naïve Bayes**, which is well-suited for text data.

---

### ⚙️ Steps Involved
1. Prepare a sample dataset of text documents.  
2. Convert text into numerical features using **CountVectorizer** or **TF-IDF**.  
3. Train a **Multinomial Naïve Bayes** model.  
4. Evaluate model accuracy on test data.  

---

### 🧠 Example: Classifying Documents into Topics (e.g., Sports, Tech, Politics)

```python
# ----------------------------------------------------------
# 📦 Import Required Libraries
# ----------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 📂 Sample Text Dataset
# ----------------------------------------------------------
data = {
    'Text': [
        'The team won the football match',
        'Election results were announced today',
        'New AI technology is emerging fast',
        'The player scored a hat-trick',
        'Government plans new policies',
        'Machine learning improves computers',
        'Cricket is a popular sport',
        'The president gave a speech',
        'Deep learning is part of AI',
        'The tennis tournament was exciting'
    ],
    'Category': [
        'Sports', 'Politics', 'Tech', 'Sports', 'Politics',
        'Tech', 'Sports', 'Politics', 'Tech', 'Sports'
    ]
}

df = pd.DataFrame(data)
print("✅ Sample Text Dataset Created!")
print(df.head())

# ----------------------------------------------------------
# 🔢 Convert Text to Numerical Features
# ----------------------------------------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Text'])
y = df['Category']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----------------------------------------------------------
# 🧠 Train the Multinomial Naïve Bayes Model
# ----------------------------------------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# ----------------------------------------------------------
# 🔍 Make Predictions
# ----------------------------------------------------------
y_pred = model.predict(X_test)

# ----------------------------------------------------------
# 📈 Evaluate Model
# ----------------------------------------------------------
print("\n✅ Naïve Bayes Text Classification Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------------------------------------
# 📊 Visualization — Confusion Matrix
# ----------------------------------------------------------
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Oranges')
plt.title('Confusion Matrix — Naïve Bayes Text Classification')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# ----------------------------------------------------------
# 🔤 Test with New Documents
# ----------------------------------------------------------
new_docs = [
    'The player broke the record',
    'New government policy announced',
    'AI improves healthcare systems'
]
new_features = vectorizer.transform(new_docs)
predictions = model.predict(new_features)

for doc, label in zip(new_docs, predictions):
    print(f"📝 Document: \"{doc}\" ➡️ Predicted Category: {label}")
```
