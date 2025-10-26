# Fake News Detection Web Application

## Overview

This is an AI-powered web application that detects fake news articles and headlines using machine learning. The system uses a PassiveAggressiveClassifier with TF-IDF vectorization to analyze text and determine authenticity. The application features a Flask backend with a Bootstrap frontend, providing real-time analysis with confidence scores.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The application follows a traditional web application architecture with the following layers:

### Backend Architecture
- **Web Framework**: Flask serves as the lightweight web framework
- **Machine Learning**: scikit-learn provides the ML capabilities with PassiveAggressiveClassifier
- **Data Processing**: pandas and numpy handle data manipulation
- **Model Persistence**: pickle for serializing trained models
- **Text Processing**: Custom TextProcessor class for NLP preprocessing

### Frontend Architecture
- **Template Engine**: Jinja2 templates with Flask
- **CSS Framework**: Bootstrap 5 with dark theme
- **Icons**: Font Awesome for UI icons
- **JavaScript**: Vanilla JavaScript for client-side interactions

## Key Components

### 1. Flask Application (app.py)
- **Purpose**: Main web server handling HTTP requests and responses
- **Key Routes**: 
  - `/` - Main page rendering
  - `/predict` - POST endpoint for news classification
- **Features**: Model loading, text processing integration, error handling

### 2. Model Training (train_model.py)
- **Purpose**: Downloads dataset and trains the ML model
- **Algorithm**: PassiveAggressiveClassifier with TF-IDF vectorization
- **Dataset**: Downloads from GitHub repository containing labeled fake/real news
- **Output**: Saves trained model and vectorizer as pickle files

### 3. Text Processing (text_processor.py)
- **Purpose**: Comprehensive text preprocessing pipeline
- **Features**: 
  - URL/email removal
  - Stop word filtering
  - Special character handling
  - Text normalization
- **Design**: Class-based approach with compiled regex patterns for efficiency

### 4. Frontend Interface (templates/index.html)
- **Purpose**: User interface for text input and result display
- **Design**: Bootstrap-based responsive design with dark theme
- **Features**: Flash message support, real-time feedback

### 5. Styling (static/style.css)
- **Purpose**: Custom styling with CSS variables and gradients
- **Theme**: Dark theme with gradient backgrounds
- **Responsive**: Mobile-first approach with Bootstrap integration

## Data Flow

1. **Training Phase**:
   - Download dataset from remote source
   - Preprocess text using TextProcessor
   - Train PassiveAggressiveClassifier with TF-IDF features
   - Save model and vectorizer to pickle files

2. **Prediction Phase**:
   - User submits news text via web form
   - Text preprocessed using TextProcessor
   - Model and vectorizer loaded from pickle files
   - Text vectorized and classified
   - Results returned with confidence score

## External Dependencies

### Python Packages
- **Flask**: Web framework for HTTP handling
- **scikit-learn**: Machine learning algorithms and vectorization
- **pandas/numpy**: Data manipulation and numerical operations
- **requests**: HTTP client for dataset downloading
- **pickle**: Model serialization (built-in)

### Frontend Dependencies
- **Bootstrap 5**: CSS framework via CDN
- **Font Awesome**: Icon library via CDN
- **Custom CSS**: Application-specific styling

### Data Source
- **Dataset**: Fake news dataset from GitHub repository
- **Format**: CSV with text and label columns
- **Source**: Public repository for training data

## Deployment Strategy

### Development Setup
- **Entry Point**: main.py runs Flask development server
- **Configuration**: Environment variables for session secret
- **Debug Mode**: Enabled for development with logging

### Model Preparation
- **Prerequisite**: Run train_model.py before starting web application
- **Model Files**: fake_news_model.pkl and tfidf_vectorizer.pkl must exist
- **Error Handling**: Graceful degradation if model files missing

### Production Considerations
- **WSGI**: Flask app ready for WSGI deployment
- **Static Files**: Served via Flask (should use nginx/Apache in production)
- **Security**: Session secret should be environment variable
- **Logging**: Configured for debugging (should be adjusted for production)

### File Structure
```
├── app.py              # Main Flask application
├── main.py             # Entry point
├── train_model.py      # Model training script
├── text_processor.py   # Text preprocessing utilities
├── templates/
│   └── index.html      # Main template
└── static/
    └── style.css       # Custom styling
```

The architecture prioritizes simplicity and modularity, making it easy to extend with additional features like user authentication, result persistence, or more sophisticated ML models.
