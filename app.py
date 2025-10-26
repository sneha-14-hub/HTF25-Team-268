import os
import logging
import pickle
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from text_processor import TextProcessor

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Create Flask application
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "fake-news-detection-secret-key")

# Initialize text processor
text_processor = TextProcessor()

def load_model():
    """Load the trained model and vectorizer"""
    try:
        with open('fake_news_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        logging.error("Model files not found. Please run train_model.py first.")
        return None, None
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None, None

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if news is fake or real"""
    try:
        # Get input text from form
        news_text = request.form.get('news_text', '').strip()
        
        if not news_text:
            flash('Please enter some news text to analyze.', 'warning')
            return redirect(url_for('index'))
        
        # Load model and vectorizer
        model, vectorizer = load_model()
        
        if model is None or vectorizer is None:
            flash('Model not available. Please train the model first by running train_model.py', 'danger')
            return redirect(url_for('index'))
        
        # Preprocess the text
        processed_text = text_processor.preprocess_text(news_text)
        
        if not processed_text.strip():
            flash('Unable to process the provided text. Please try with different content.', 'warning')
            return redirect(url_for('index'))
        
        # Transform text using the trained vectorizer
        text_vector = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(text_vector)[0]
        confidence = model.predict_proba(text_vector)[0]
        
        # Get confidence score for the predicted class
        confidence_score = max(confidence) * 100
        
        # Determine result
        if prediction == 1:  # Real news
            result = {
                'is_real': True,
                'label': '✅ Real News',
                'confidence': confidence_score,
                'class': 'success'
            }
        else:  # Fake news
            result = {
                'is_real': False,
                'label': '❌ Fake News',
                'confidence': confidence_score,
                'class': 'danger'
            }
        
        return render_template('index.html', 
                             result=result, 
                             input_text=news_text)
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        flash(f'An error occurred during prediction: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model, vectorizer = load_model()
    status = "healthy" if model and vectorizer else "model_not_loaded"
    return jsonify({"status": status})

if __name__ == '__main__':
    # Check if model exists
    model, vectorizer = load_model()
    if model is None:
        print("\n" + "="*60)
        print("⚠️  MODEL NOT FOUND!")
        print("Please run 'python train_model.py' to train the model first.")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("✅ Model loaded successfully!")
        print("🚀 Starting Fake News Detection App...")
        print("📱 Access the app at: http://localhost:5000")
        print("="*60 + "\n")
    
    # Start the Flask application
    app.run(host='0.0.0.0', port=5000, debug=True)
