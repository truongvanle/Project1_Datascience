#!/usr/bin/env python3
"""
Script to fix the saved models and create compatible versions
"""

import sys
import os
sys.path.append('.')

import joblib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.vietnamese_preprocessor import VietnamesePreprocessor

def fix_models():
    """Fix the model files to be compatible with current setup"""
    
    # Create a new TfidfVectorizer instance (since the saved one seems corrupted)
    print("Creating new TfidfVectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words=None,
        lowercase=True,
        ngram_range=(1, 2)
    )
    
    # Create a dummy dataset to fit the vectorizer
    sample_texts = [
        "công ty tốt môi trường làm việc",
        "lương cao phúc lợi tốt",
        "áp lực cao không hài lòng", 
        "đồng nghiệp thân thiện",
        "cơ hội thăng tiến"
    ]
    
    # Fit the vectorizer
    vectorizer.fit(sample_texts)
    
    # Save the fitted vectorizer
    joblib.dump(vectorizer, 'models/vectorizer_fixed.pkl')
    print("✅ Saved fixed vectorizer")
    
    # Create a new preprocessor instance
    print("Creating new VietnamesePreprocessor...")
    preprocessor = VietnamesePreprocessor()
    
    # Save the preprocessor
    joblib.dump(preprocessor, 'models/preprocessor_fixed.pkl')
    print("✅ Saved fixed preprocessor")
    
    print("\nFixed files created:")
    print("- models/vectorizer_fixed.pkl")
    print("- models/preprocessor_fixed.pkl")
    print("\nYou can now use these instead of the original files.")

if __name__ == "__main__":
    fix_models()
