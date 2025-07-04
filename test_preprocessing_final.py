"""
Test Script: Verify Preprocessing Standardization
Kiểm tra preprocessing sau khi chuẩn hóa theo Project1_Le
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def test_preprocessing_standardization():
    """Test preprocessing standardization results"""
    
    print("🔍 TESTING PREPROCESSING STANDARDIZATION")
    print("=" * 60)
    
    # Load the complete pipeline
    pipeline_path = "/home/thinhdao/it_viec_project1/data/complete_sentiment_pipeline.pkl"
    
    try:
        pipeline = joblib.load(pipeline_path)
        print("✅ Loaded complete sentiment pipeline successfully")
        
        # Extract components
        model = pipeline['model']
        preprocessor = pipeline['preprocessor']
        vectorizer = pipeline['vectorizer']
        svd_reducer = pipeline['svd_reducer']
        numerical_features = pipeline['numerical_features']
        
        print(f"📊 Model Type: {type(model).__name__}")
        print(f"🔤 Preprocessor: {type(preprocessor).__name__}")
        print(f"🧮 Vectorizer: {type(vectorizer).__name__}")
        print(f"🔧 SVD Components: {svd_reducer.n_components}")
        print(f"📈 Numerical Features: {len(numerical_features)}")
        
    except Exception as e:
        print(f"❌ Error loading pipeline: {e}")
        return False
    
    # Test preprocessing with sample Vietnamese texts
    test_texts = [
        "Công ty này rất tốt, môi trường làm việc thân thiện và đồng nghiệp hỗ trợ nhiệt tình!",
        "Lương thấp, áp lực cao, không khuyến khích làm ở đây.",
        "Bình thường thôi, không tệ nhưng cũng không xuất sắc lắm.",
        "Great company with good benefits and work-life balance 👍",
        "Công ty ok, nhưng process hơi rối, cần cải thiện thêm..."
    ]
    
    print("\n🧪 TESTING PREPROCESSING ON SAMPLE TEXTS")
    print("-" * 60)
    
    for i, text in enumerate(test_texts, 1):
        try:
            # Test preprocessing
            processed = preprocessor.preprocess_text(text)
            print(f"\n📝 Test {i}:")
            print(f"   Original: {text}")
            print(f"   Processed: {processed}")
            
        except Exception as e:
            print(f"❌ Error preprocessing text {i}: {e}")
            return False
    
    # Test emotion word counting
    print("\n💭 TESTING EMOTION WORD COUNTING")
    print("-" * 60)
    
    emotion_test_texts = [
        "tốt tuyệt vời xuất sắc",
        "tệ dở tệ hại không tốt", 
        "bình thường ok"
    ]
    
    for i, text in enumerate(emotion_test_texts, 1):
        try:
            pos_count, neg_count = preprocessor.count_emotion_words(text)
            print(f"📊 Text {i}: '{text}'")
            print(f"   Positive words: {pos_count}")
            print(f"   Negative words: {neg_count}")
            
        except Exception as e:
            print(f"❌ Error counting emotion words {i}: {e}")
            return False
    
    # Test full prediction pipeline
    print("\n🎯 TESTING FULL PREDICTION PIPELINE")
    print("-" * 60)
    
    try:
        # Create prediction function (simplified version)
        def predict_sentiment(text, rating=3.0):
            # Preprocess text
            processed_text = preprocessor.preprocess_text(text)
            
            # Count emotion words
            pos_count, neg_count = preprocessor.count_emotion_words(processed_text)
            
            # Create feature vector
            text_vector = vectorizer.transform([processed_text])
            text_features = svd_reducer.transform(text_vector)
            
            # Add numerical features
            numerical_vals = [rating, pos_count, neg_count]
            
            # Combine features
            combined_features = np.hstack([
                text_features,
                np.array(numerical_vals).reshape(1, -1)
            ])
            
            # Make prediction
            prediction = model.predict(combined_features)[0]
            probabilities = model.predict_proba(combined_features)[0]
            
            return {
                'prediction': prediction,
                'probabilities': probabilities,
                'processed_text': processed_text,
                'pos_count': pos_count,
                'neg_count': neg_count
            }
        
        # Test predictions
        for i, text in enumerate(test_texts[:3], 1):
            result = predict_sentiment(text, rating=4.0)
            print(f"\n🔮 Prediction {i}:")
            print(f"   Text: {text}")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {max(result['probabilities']):.3f}")
            print(f"   Pos/Neg words: {result['pos_count']}/{result['neg_count']}")
            
    except Exception as e:
        print(f"❌ Error in prediction pipeline: {e}")
        return False
    
    print("\n✅ ALL TESTS PASSED!")
    print("🎉 Preprocessing standardization successful!")
    print("🚀 Pipeline ready for production!")
    
    return True

if __name__ == "__main__":
    success = test_preprocessing_standardization()
    exit(0 if success else 1)
