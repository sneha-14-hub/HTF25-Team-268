import pandas as pd
import re
import nltk
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords
nltk.download("stopwords")

# Load fake and true news datasets
fake_df = pd.read_csv("fake.csv")
true_df = pd.read_csv("true.csv")

# Add labels: 0 for fake news, 1 for true news
fake_df["label"] = 0
true_df["label"] = 1

# Assuming both have a column named 'text' or 'content', adjust accordingly
# If your CSVs have different column names, replace 'text' with the correct one
fake_df.rename(columns={fake_df.columns[0]: "content"}, inplace=True)
true_df.rename(columns={true_df.columns[0]: "content"}, inplace=True)

# Combine datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Preprocessing
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

df.dropna(subset=["content", "label"], inplace=True)
df["content"] = df["content"].apply(clean_text)

# Feature and label split
X = df["content"]
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Test model
y_pred = model.predict(X_test_vec)
print("Model Performance:\n")
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
print("\n✅ Model and vectorizer saved successfully!")