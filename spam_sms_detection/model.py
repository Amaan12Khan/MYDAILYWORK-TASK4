import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# =====================
# 1. LOAD DATA
# =====================
print("📂 Loading dataset...")
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels → 0 & 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print("✅ Dataset loaded")

# =====================
# 2. TEXT PROCESSING
# =====================
vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(df['message'])
y = df['label']

# =====================
# 3. SPLIT DATA
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================
# 4. TRAIN MODEL
# =====================
model = MultinomialNB()
model.fit(X_train, y_train)

# =====================
# 5. EVALUATION
# =====================
y_pred = model.predict(X_test)

print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📄 Report:\n", classification_report(y_test, y_pred))

# =====================
# 6. SAVE MODEL
# =====================
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n💾 Model & vectorizer saved!")