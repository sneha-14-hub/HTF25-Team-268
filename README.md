# Fake News Detection Web Application

An AI-powered web application that uses machine learning to detect fake news articles and headlines. Built with Flask, scikit-learn, and Bootstrap for a modern, responsive interface.

## Features

- **Machine Learning Classification**: Uses PassiveAggressiveClassifier with TF-IDF vectorization
- **Real Dataset Training**: Trains on authentic fake news datasets
- **Web Interface**: Clean, responsive Bootstrap interface
- **Real-time Analysis**: Instant classification with confidence scores
- **Text Preprocessing**: Advanced NLP preprocessing pipeline
- **Model Persistence**: Saves trained models for reuse

## Tech Stack

**Backend:**
- Flask (Web framework)
- scikit-learn (Machine learning)
- pandas & numpy (Data processing)
- pickle (Model serialization)

**Frontend:**
- HTML5 & Bootstrap 5
- Font Awesome icons
- Custom CSS with dark theme
- Vanilla JavaScript

## Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Step 1: Install Dependencies
The application will automatically handle dependencies, but make sure you have Python installed.

### Step 2: Train the Model
Before running the web application, you need to train the machine learning model:

```bash
python train_model.py



