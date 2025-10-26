"""
Fake News Detection Model Training Script

This script downloads a fake news dataset, preprocesses the data,
trains a machine learning model, and saves it for use in the web application.
"""

import os
import logging
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import requests
from io import StringIO
from text_processor import TextProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FakeNewsModelTrainer:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.model = None
        self.vectorizer = None
        
    def download_dataset(self):
        """Download fake news dataset"""
        logging.info("Downloading fake news dataset...")
        
        # Using a publicly available fake news dataset
        # This is a sample dataset URL - in production, you'd use a more comprehensive dataset
        dataset_url = "https://raw.githubusercontent.com/nishitpatel01/Fake_News_Detection/main/train.csv"
        
        try:
            response = requests.get(dataset_url, timeout=30)
            response.raise_for_status()
            
            # Save the dataset
            with open('fake_news_dataset.csv', 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            logging.info("Dataset downloaded successfully!")
            return True
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download dataset: {str(e)}")
            
            # Create a sample dataset for demonstration if download fails
            logging.info("Creating sample dataset for demonstration...")
            self.create_sample_dataset()
            return True
    
    def create_sample_dataset(self):
        """Create a sample dataset for demonstration purposes"""
        sample_data = {
            'title': [
                "Scientists discover new species of butterfly in Amazon rainforest",
                "BREAKING: Aliens land in New York City, demand pizza",
                "Local community center opens new after-school program",
                "Miracle cure found in grandmother's backyard herbs",
                "University researchers publish study on climate change effects",
                "Government secretly controls weather with hidden machines",
                "New transportation bill passes senate with bipartisan support",
                "Celebrity spotted eating regular food like normal person",
                "Healthcare workers receive recognition for pandemic efforts",
                "Ancient aliens built pyramids using time travel technology"
            ] * 100,  # Multiply to have enough data
            'text': [
                "Researchers from the University of Science have documented a new butterfly species...",
                "In an unprecedented event, extraterrestrial visitors landed in Times Square...",
                "The Riverside Community Center announced the opening of its new program...",
                "A local woman claims her grandmother's herb garden contains the secret...",
                "A comprehensive study published in Nature Climate reveals significant...",
                "Secret documents allegedly prove that the government has been...",
                "The Senate voted 67-33 to approve the new infrastructure bill...",
                "Photos emerged showing the famous actor eating a sandwich...",
                "Medical professionals across the country were honored for their...",
                "Conspiracy theorists claim that ancient structures were built..."
            ] * 100,
            'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 100  # 1 = Real, 0 = Fake
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv('fake_news_dataset.csv', index=False)
        logging.info("Sample dataset created successfully!")
    
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        logging.info("Loading and preprocessing dataset...")
        
        try:
            # Load dataset
            df = pd.read_csv('fake_news_dataset.csv')
            
            # Check if required columns exist
            if 'title' not in df.columns or 'label' not in df.columns:
                # Try alternative column names
                if 'text' in df.columns:
                    df['title'] = df['text']
                elif 'headline' in df.columns:
                    df['title'] = df['headline']
                else:
                    raise ValueError("Dataset must contain 'title' or 'text' column")
            
            # Handle missing values
            df = df.dropna(subset=['title', 'label'])
            
            # Combine title and text if both exist
            if 'text' in df.columns:
                df['combined_text'] = df['title'].astype(str) + " " + df['text'].astype(str)
            else:
                df['combined_text'] = df['title'].astype(str)
            
            # Preprocess text
            logging.info("Preprocessing text data...")
            df['processed_text'] = df['combined_text'].apply(self.text_processor.preprocess_text)
            
            # Remove empty processed texts
            df = df[df['processed_text'].str.strip() != '']
            
            # Ensure labels are binary (0 for fake, 1 for real)
            df['label'] = df['label'].astype(int)
            
            logging.info(f"Dataset loaded successfully! Shape: {df.shape}")
            logging.info(f"Class distribution:\n{df['label'].value_counts()}")
            
            return df['processed_text'].values, df['label'].values
            
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            raise
    
    def train_model(self, texts, labels):
        """Train the fake news detection model"""
        logging.info("Training the model...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            max_df=0.95,
            min_df=2
        )
        
        # Fit and transform the training data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train LogisticRegression for binary classification with probability prediction
        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            solver='liblinear'  # Good for small datasets
        )
        
        self.model.fit(X_train_tfidf, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_tfidf)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f"Model training completed!")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        return accuracy
    
    def save_model(self):
        """Save the trained model and vectorizer"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained yet!")
        
        # Save model
        with open('fake_news_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save vectorizer
        with open('tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        logging.info("Model and vectorizer saved successfully!")
    
    def test_model(self):
        """Test the model with sample predictions"""
        logging.info("Testing model with sample texts...")
        
        test_texts = [
            "Scientists at MIT have developed a new breakthrough in renewable energy technology",
            "BREAKING: Celebrity spotted eating food like a normal human being",
            "Local government announces new infrastructure improvements for downtown area",
            "Miracle weight loss pill melts fat overnight without diet or exercise"
        ]
        
        expected_labels = ["REAL", "FAKE", "REAL", "FAKE"]
        
        for i, text in enumerate(test_texts):
            processed_text = self.text_processor.preprocess_text(text)
            text_vector = self.vectorizer.transform([processed_text])
            prediction = self.model.predict(text_vector)[0]
            confidence = self.model.predict_proba(text_vector)[0]
            
            result = "REAL" if prediction == 1 else "FAKE"
            conf_score = max(confidence) * 100
            
            logging.info(f"Text: {text[:50]}...")
            logging.info(f"Predicted: {result} (Expected: {expected_labels[i]}) - Confidence: {conf_score:.2f}%")
            logging.info("-" * 50)

def main():
    """Main training function"""
    print("=" * 60)
    print("🤖 FAKE NEWS DETECTION MODEL TRAINER")
    print("=" * 60)
    
    trainer = FakeNewsModelTrainer()
    
    try:
        # Step 1: Download dataset
        trainer.download_dataset()
        
        # Step 2: Load and preprocess data
        texts, labels = trainer.load_and_preprocess_data()
        
        # Step 3: Train model
        accuracy = trainer.train_model(texts, labels)
        
        # Step 4: Save model
        trainer.save_model()
        
        # Step 5: Test model
        trainer.test_model()
        
        print("\n" + "=" * 60)
        print(f"✅ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📊 Model Accuracy: {accuracy:.4f}")
        print(f"💾 Model saved as: fake_news_model.pkl")
        print(f"💾 Vectorizer saved as: tfidf_vectorizer.pkl")
        print("🚀 You can now run the web application with: python app.py")
        print("=" * 60)
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        print(f"\n❌ Training failed: {str(e)}")
        print("Please check the logs above for more details.")

if __name__ == "__main__":
    main()
