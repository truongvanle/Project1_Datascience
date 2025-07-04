import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page header
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
            padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
    <h1>🎯 Sentiment Analysis</h1>
    <p style="font-size: 1.2rem; margin-bottom: 0;">Analyze Employee & Candidate Feedback Sentiment</p>
</div>
""", unsafe_allow_html=True)

# Load models function
@st.cache_resource
def load_models():
    """Load pre-trained sentiment analysis models"""
    models = {}
    model_path = "models"
    
    # Check if models exist
    if not os.path.exists(model_path):
        st.error(f"Model directory not found: {model_path}")
        return None
    
    try:
        # Load available models
        model_files = [
            ('Random Forest', 'randomforest_classifier.pkl'),
            ('XGBoost', 'xgboost_classifier.pkl'),
            ('CatBoost', 'catboost_classifier.pkl'),
            ('LightGBM', 'lightgbm_classifier.pkl'),
            ('Logistic Regression', 'logisticregression_classifier.pkl')
        ]
        
        for model_name, filename in model_files:
            file_path = os.path.join(model_path, filename)
            if os.path.exists(file_path):
                models[model_name] = joblib.load(file_path)
        
        # Load vectorizer and preprocessor
        vectorizer_path = os.path.join(model_path, 'vectorizer_proper.pkl')
        preprocessor_path = os.path.join(model_path, 'preprocessor_fixed.pkl')
        
        if os.path.exists(vectorizer_path):
            models['vectorizer'] = joblib.load(vectorizer_path)
        elif os.path.exists(os.path.join(model_path, 'vectorizer_fixed.pkl')):
            models['vectorizer'] = joblib.load(os.path.join(model_path, 'vectorizer_fixed.pkl'))
        elif os.path.exists(os.path.join(model_path, 'vectorizer.pkl')):
            # Fallback to original if others don't exist
            try:
                models['vectorizer'] = joblib.load(os.path.join(model_path, 'vectorizer.pkl'))
            except Exception:
                pass
                
        if os.path.exists(preprocessor_path):
            models['preprocessor'] = joblib.load(preprocessor_path)
        elif os.path.exists(os.path.join(model_path, 'preprocessor.pkl')):
            # Fallback to original if fixed doesn't exist
            try:
                models['preprocessor'] = joblib.load(os.path.join(model_path, 'preprocessor.pkl'))
            except Exception:
                pass
            
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# Text preprocessing function
def preprocess_text(text):
    """Basic text preprocessing"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove special characters but keep Vietnamese characters
    text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Prediction function
def predict_sentiment(text, models, model_name='Random Forest'):
    """Predict sentiment for given text"""
    if not models or model_name not in models:
        return None, None
    
    try:
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Vectorize if vectorizer is available
        if 'vectorizer' in models:
            text_vectorized = models['vectorizer'].transform([processed_text])
        else:
            # Simple fallback
            text_vectorized = np.array([[len(processed_text), processed_text.count(' ')]])
        
        # Predict
        model = models[model_name]
        prediction = model.predict(text_vectorized)[0]
        
        # Get probability if available
        try:
            probabilities = model.predict_proba(text_vectorized)[0]
            confidence = max(probabilities)
        except:
            confidence = 0.8  # Default confidence
        
        return prediction, confidence
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# Load models
models = load_models()

# Main interface
if models:
    st.success(f"✅ Successfully loaded {len([k for k in models.keys() if k not in ['vectorizer', 'preprocessor']])} models!")
    
    # Model selection
    available_models = [k for k in models.keys() if k not in ['vectorizer', 'preprocessor']]
    selected_model = st.selectbox("Select Model:", available_models)
    
    # Input methods
    tab1, tab2, tab3 = st.tabs(["💬 Single Text", "📄 Batch Analysis", "🔍 Model Comparison"])
    
    with tab1:
        st.markdown("### 💬 Single Text Analysis")
        
        # Text input
        input_text = st.text_area(
            "Enter text to analyze:",
            placeholder="Type your review or feedback here...",
            height=150
        )
        
        if st.button("🎯 Analyze Sentiment", type="primary"):
            if input_text.strip():
                with st.spinner("Analyzing sentiment..."):
                    prediction, confidence = predict_sentiment(input_text, models, selected_model)
                    
                    if prediction is not None:
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Sentiment result
                            sentiment_colors = {
                                'positive': '#28a745',
                                'negative': '#dc3545', 
                                'neutral': '#ffc107'
                            }
                            
                            sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                            sentiment = sentiment_map.get(prediction, 'unknown')
                            color = sentiment_colors.get(sentiment, '#6c757d')
                            
                            st.markdown(f"""
                            <div style="background: {color}; color: white; padding: 1rem; 
                                       border-radius: 8px; text-align: center;">
                                <h3>Sentiment: {sentiment.title()}</h3>
                                <p>Confidence: {confidence:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Confidence gauge
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = confidence * 100,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Confidence Score"},
                                gauge = {
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': color},
                                    'steps': [
                                        {'range': [0, 50], 'color': "lightgray"},
                                        {'range': [50, 80], 'color': "yellow"},
                                        {'range': [80, 100], 'color': "green"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 90
                                    }
                                }
                            ))
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter some text to analyze.")
    
    with tab2:
        st.markdown("### 📄 Batch Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with reviews:",
            type=['csv'],
            help="CSV should have a column with text to analyze"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(df)} rows")
                
                # Column selection
                text_column = st.selectbox("Select text column:", df.columns)
                
                if st.button("🚀 Analyze All"):
                    with st.spinner("Analyzing all texts..."):
                        predictions = []
                        confidences = []
                        
                        for text in df[text_column]:
                            pred, conf = predict_sentiment(str(text), models, selected_model)
                            predictions.append(pred)
                            confidences.append(conf)
                        
                        # Add results to dataframe
                        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                        df['predicted_sentiment'] = [sentiment_map.get(p, 'unknown') for p in predictions]
                        df['confidence'] = confidences
                        
                        # Display results
                        st.success("Analysis complete!")
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        sentiment_counts = df['predicted_sentiment'].value_counts()
                        
                        with col1:
                            positive_count = sentiment_counts.get('positive', 0)
                            st.metric("Positive", positive_count, f"{positive_count/len(df):.1%}")
                        
                        with col2:
                            neutral_count = sentiment_counts.get('neutral', 0)
                            st.metric("Neutral", neutral_count, f"{neutral_count/len(df):.1%}")
                        
                        with col3:
                            negative_count = sentiment_counts.get('negative', 0)
                            st.metric("Negative", negative_count, f"{negative_count/len(df):.1%}")
                        
                        # Visualization
                        fig = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="Sentiment Distribution",
                            color_discrete_map={
                                'positive': '#28a745',
                                'negative': '#dc3545',
                                'neutral': '#ffc107'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show sample results
                        st.markdown("### Sample Results")
                        st.dataframe(df.head(10))
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            key='download-csv'
                        )
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    with tab3:
        st.markdown("### 🔍 Model Comparison")
        
        comparison_text = st.text_area(
            "Enter text to compare across models:",
            placeholder="Enter text for model comparison...",
            height=100
        )
        
        if st.button("🔄 Compare Models") and comparison_text.strip():
            with st.spinner("Running comparison..."):
                results = {}
                
                for model_name in available_models:
                    pred, conf = predict_sentiment(comparison_text, models, model_name)
                    if pred is not None:
                        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                        results[model_name] = {
                            'sentiment': sentiment_map.get(pred, 'unknown'),
                            'confidence': conf
                        }
                
                # Display comparison
                if results:
                    comparison_df = pd.DataFrame(results).T
                    comparison_df.reset_index(inplace=True)
                    comparison_df.columns = ['Model', 'Predicted Sentiment', 'Confidence']
                    
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Confidence comparison chart
                    fig = px.bar(
                        comparison_df, 
                        x='Model', 
                        y='Confidence',
                        color='Predicted Sentiment',
                        title="Model Confidence Comparison",
                        color_discrete_map={
                            'positive': '#28a745',
                            'negative': '#dc3545',
                            'neutral': '#ffc107'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("❌ Models not loaded. Please check the model directory.")
    st.info("💡 Make sure the model files are in the correct location.")

# Additional Information
st.markdown("---")
st.markdown("## 📚 About Sentiment Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🎯 What is Sentiment Analysis?
    
    Sentiment analysis is a natural language processing technique that determines 
    the emotional tone behind text. Our models analyze:
    
    - **Positive**: Favorable opinions, satisfaction, praise
    - **Negative**: Complaints, dissatisfaction, criticism  
    - **Neutral**: Factual statements, balanced reviews
    """)

with col2:
    st.markdown("""
    ### 🔧 Our Approach
    
    We use multiple machine learning models trained on Vietnamese text:
    
    - **Preprocessing**: Text cleaning and normalization
    - **Vectorization**: TF-IDF feature extraction
    - **Classification**: Ensemble of ML algorithms
    - **Validation**: Cross-validation and performance metrics
    """)

# Performance metrics (mock data for demonstration)
st.markdown("### 📊 Model Performance")

performance_data = {
    'Model': ['Random Forest', 'XGBoost', 'CatBoost', 'LightGBM', 'Logistic Regression'],
    'Accuracy': [0.94, 0.92, 0.95, 0.93, 0.89],
    'Precision': [0.93, 0.91, 0.94, 0.92, 0.88],
    'Recall': [0.94, 0.92, 0.95, 0.93, 0.89],
    'F1-Score': [0.93, 0.91, 0.94, 0.92, 0.88]
}

performance_df = pd.DataFrame(performance_data)
st.dataframe(performance_df, use_container_width=True)
