"""
Text Processing Utilities for Fake News Detection

This module contains functions for preprocessing text data,
including cleaning, tokenization, and feature extraction.
"""

import re
import string
import logging
from typing import List, Optional

class TextProcessor:
    """Text preprocessing utilities for fake news detection"""
    
    def __init__(self):
        # Common English stop words (basic set)
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more',
            'very', 'when', 'come', 'may', 'its', 'only', 'think', 'now', 'work',
            'life', 'become', 'here', 'how', 'after', 'back', 'other', 'many',
            'than', 'first', 'been', 'way', 'who', 'oil', 'sit', 'now', 'find',
            'long', 'down', 'day', 'did', 'get', 'has', 'his', 'had', 'let',
            'put', 'too', 'old', 'any', 'app', 'may', 'new', 'try', 'us', 'man',
            'over', 'think', 'also', 'your', 'work', 'life', 'only', 'can',
            'still', 'should', 'after', 'being', 'now', 'made', 'before', 'here',
            'through', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can',
            'will', 'just', 'dont', 'should', 'now'
        }
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.number_pattern = re.compile(r'\d+')
        self.whitespace_pattern = re.compile(r'\s+')
        self.special_chars_pattern = re.compile(r'[^a-zA-Z\s]')
        
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        return self.url_pattern.sub('', text)
    
    def remove_emails(self, text: str) -> str:
        """Remove email addresses from text"""
        return self.email_pattern.sub('', text)
    
    def remove_mentions(self, text: str) -> str:
        """Remove social media mentions (@username)"""
        return self.mention_pattern.sub('', text)
    
    def remove_hashtags(self, text: str) -> str:
        """Remove hashtags but keep the text"""
        return self.hashtag_pattern.sub(lambda m: m.group(0)[1:], text)
    
    def remove_numbers(self, text: str) -> str:
        """Remove numbers from text"""
        return self.number_pattern.sub('', text)
    
    def remove_special_characters(self, text: str) -> str:
        """Remove special characters and punctuation"""
        return self.special_chars_pattern.sub(' ', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace (multiple spaces to single space)"""
        return self.whitespace_pattern.sub(' ', text).strip()
    
    def remove_stop_words(self, text: str) -> str:
        """Remove common stop words"""
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing unwanted characters and formatting
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = self.remove_urls(text)
        
        # Remove emails
        text = self.remove_emails(text)
        
        # Remove mentions
        text = self.remove_mentions(text)
        
        # Remove hashtags (keep text part)
        text = self.remove_hashtags(text)
        
        # Remove numbers
        text = self.remove_numbers(text)
        
        # Remove special characters and punctuation
        text = self.remove_special_characters(text)
        
        # Normalize whitespace
        text = self.normalize_whitespace(text)
        
        return text
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete text preprocessing pipeline
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Preprocessed text ready for model input
        """
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # Step 1: Clean the text
            cleaned_text = self.clean_text(text)
            
            # Step 2: Remove stop words
            processed_text = self.remove_stop_words(cleaned_text)
            
            # Step 3: Final cleanup
            processed_text = self.normalize_whitespace(processed_text)
            
            return processed_text
            
        except Exception as e:
            logging.error(f"Error preprocessing text: {str(e)}")
            return ""
    
    def extract_features(self, text: str) -> dict:
        """
        Extract additional features from text that might be useful for classification
        
        Args:
            text (str): Text to extract features from
            
        Returns:
            dict: Dictionary of extracted features
        """
        if not text or not isinstance(text, str):
            return {}
        
        features = {}
        
        try:
            # Basic text statistics
            features['char_count'] = len(text)
            features['word_count'] = len(text.split())
            features['sentence_count'] = len(text.split('.'))
            features['avg_word_length'] = sum(len(word) for word in text.split()) / max(len(text.split()), 1)
            
            # Count special elements
            features['url_count'] = len(self.url_pattern.findall(text))
            features['email_count'] = len(self.email_pattern.findall(text))
            features['mention_count'] = len(self.mention_pattern.findall(text))
            features['hashtag_count'] = len(self.hashtag_pattern.findall(text))
            features['number_count'] = len(self.number_pattern.findall(text))
            
            # Count uppercase words (might indicate shouting/emphasis)
            words = text.split()
            features['uppercase_word_count'] = sum(1 for word in words if word.isupper())
            features['uppercase_ratio'] = features['uppercase_word_count'] / max(len(words), 1)
            
            # Count exclamation marks and question marks
            features['exclamation_count'] = text.count('!')
            features['question_count'] = text.count('?')
            
            # Count punctuation
            features['punctuation_count'] = sum(1 for char in text if char in string.punctuation)
            features['punctuation_ratio'] = features['punctuation_count'] / max(len(text), 1)
            
        except Exception as e:
            logging.error(f"Error extracting features: {str(e)}")
        
        return features
    
    def validate_text(self, text: str, min_length: int = 10, max_length: int = 10000) -> tuple:
        """
        Validate if text is suitable for processing
        
        Args:
            text (str): Text to validate
            min_length (int): Minimum required length
            max_length (int): Maximum allowed length
            
        Returns:
            tuple: (is_valid: bool, error_message: str)
        """
        if not text or not isinstance(text, str):
            return False, "Text is empty or not a string"
        
        text = text.strip()
        
        if len(text) < min_length:
            return False, f"Text is too short (minimum {min_length} characters required)"
        
        if len(text) > max_length:
            return False, f"Text is too long (maximum {max_length} characters allowed)"
        
        # Check if text contains meaningful content (not just special characters)
        cleaned = self.clean_text(text)
        if len(cleaned.strip()) < 5:
            return False, "Text does not contain enough meaningful content"
        
        return True, ""

# Utility functions for backward compatibility
def preprocess_text(text: str) -> str:
    """Standalone function for text preprocessing"""
    processor = TextProcessor()
    return processor.preprocess_text(text)

def clean_text(text: str) -> str:
    """Standalone function for text cleaning"""
    processor = TextProcessor()
    return processor.clean_text(text)
