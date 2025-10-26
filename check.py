import joblib
from preprocess import clean_text
from newspaper import Article

model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def predict_text(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return "FAKE" if prediction[0] == 0 else "REAL"

def predict_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return predict_text(article.text)
    except:
        return "Error: Could not process the URL."