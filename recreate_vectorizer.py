#!/usr/bin/env python3
"""
Recreate the vectorizer using the actual training data
"""

import sys
import os
sys.path.append('.')

import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.vietnamese_preprocessor import VietnamesePreprocessor

def preprocess_text(text):
    """Basic text preprocessing to match the model training"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove special characters but keep Vietnamese characters
    text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def recreate_vectorizer():
    """Recreate the vectorizer using actual training data"""
    
    print("Loading training data...")
    # Load the data
    try:
        df = pd.read_csv('data/reviews.csv')
        print(f"✅ Loaded {len(df)} reviews")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Combine text columns
    text_columns = ['Title', 'What I liked', 'Suggestions for improvement']
    
    # Combine all text into a single corpus
    all_texts = []
    for _, row in df.iterrows():
        combined_text = ""
        for col in text_columns:
            if not pd.isna(row[col]):
                combined_text += str(row[col]) + " "
        
        if combined_text.strip():
            processed_text = preprocess_text(combined_text.strip())
            if processed_text:
                all_texts.append(processed_text)
    
    print(f"✅ Prepared {len(all_texts)} text samples")
    print(f"Sample text: {all_texts[0][:100]}...")
    
    # Create and fit vectorizer with parameters that likely match the original
    print("Creating vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=1046,  # Match the expected feature count
        stop_words=None,
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    # Fit the vectorizer
    print("Fitting vectorizer on training data...")
    vectorizer.fit(all_texts)
    
    print(f"✅ Vectorizer fitted with {len(vectorizer.get_feature_names_out())} features")
    
    # Save the vectorizer
    joblib.dump(vectorizer, 'models/vectorizer_proper.pkl')
    print("✅ Saved vectorizer_proper.pkl")
    
    # Test the vectorizer
    test_text = "Công ty này rất tốt và môi trường làm việc tuyệt vời"
    processed_test = preprocess_text(test_text)
    vectorized = vectorizer.transform([processed_test])
    print(f"✅ Test vectorization: {vectorized.shape} features")
    
    # Test with a model
    print("Testing with a classifier...")
    try:
        model = joblib.load('models/randomforest_classifier.pkl')
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0].max()
        print(f"✅ Test prediction: {prediction}, Probability: {probability:.3f}")
    except Exception as e:
        print(f"❌ Model test failed: {e}")
    
    return vectorizer

if __name__ == "__main__":
    recreate_vectorizer()
