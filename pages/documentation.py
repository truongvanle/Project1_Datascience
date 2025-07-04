import streamlit as st
import pandas as pd
import os
from datetime import datetime

# Page header
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
            padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
    <h1>📚 Documentation</h1>
    <p style="font-size: 1.2rem; margin-bottom: 0;">Technical Documentation & Resources</p>
</div>
""", unsafe_allow_html=True)

# Main documentation tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖 User Guide", 
    "🔧 Technical Docs", 
    "📓 Jupyter Notebooks", 
    "🌐 API Reference",
    "❓ FAQ"
])

with tab1:
    st.markdown("### 📖 User Guide")
    
    # Getting started section
    st.markdown("#### 🚀 Getting Started")
    
    st.markdown("""
    Welcome to the ITViec Analytics Platform! This comprehensive guide will help you navigate 
    and utilize all features of our sentiment analysis and clustering system.
    
    ##### Quick Start Steps:
    1. **📊 Business Overview**: Start here to understand project objectives
    2. **🎯 Sentiment Analysis**: Try predicting sentiment on sample text
    3. **🔍 Information Clustering**: Explore review clustering and patterns
    4. **📈 Data Exploration**: Examine the underlying dataset
    5. **🔬 Model Training**: Learn about our ML models and training process
    """)
    
    # Feature overview
    st.markdown("#### ✨ Features Overview")
    
    features = [
        {
            "Feature": "🎯 Sentiment Analysis",
            "Description": "Analyze text sentiment as positive, negative, or neutral",
            "Use Cases": "Employee feedback analysis, company reputation monitoring",
            "How to Use": "Navigate to Sentiment Analysis tab, enter text, click Analyze"
        },
        {
            "Feature": "🔍 Information Clustering", 
            "Description": "Group similar reviews to identify patterns and themes",
            "Use Cases": "Topic discovery, pattern recognition, improvement areas",
            "How to Use": "Go to Clustering tab, select method (K-Means/LDA), run analysis"
        },
        {
            "Feature": "📈 Data Exploration",
            "Description": "Interactive analysis of the ITViec reviews dataset",
            "Use Cases": "Data insights, statistical analysis, trend identification",
            "How to Use": "Visit Data Exploration for charts and statistics"
        },
        {
            "Feature": "🔬 Model Training",
            "Description": "View ML model performance and training details",
            "Use Cases": "Understanding model accuracy, comparing algorithms",
            "How to Use": "Check Model Training tab for performance metrics"
        }
    ]
    
    features_df = pd.DataFrame(features)
    
    for _, feature in features_df.iterrows():
        with st.expander(f"{feature['Feature']}"):
            st.write(f"**Description:** {feature['Description']}")
            st.write(f"**Use Cases:** {feature['Use Cases']}")
            st.write(f"**How to Use:** {feature['How to Use']}")
    
    # Best practices
    st.markdown("#### 💡 Best Practices")
    
    st.markdown("""
    **For Sentiment Analysis:**
    - Use clear, complete sentences for better accuracy
    - Vietnamese text is fully supported with Underthesea processing
    - Longer reviews (50+ words) typically yield more accurate results
    - Compare multiple models to understand prediction confidence
    
    **For Clustering Analysis:**
    - Start with 3-5 clusters for initial exploration
    - Review cluster keywords to understand themes
    - Use word clouds to visualize cluster characteristics
    - Analyze cluster ratings to identify improvement areas
    
    **For Data Exploration:**
    - Check data completeness before analysis
    - Look for temporal patterns in review submissions
    - Compare ratings across different companies
    - Identify outliers and unusual patterns
    """)

with tab2:
    st.markdown("### 🔧 Technical Documentation")
    
    # Architecture overview
    st.markdown("#### 🏗️ System Architecture")
    
    st.markdown("""
    **Application Architecture:**
    ```
    ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
    │   Streamlit     │    │   ML Models      │    │   Data Layer    │
    │   Frontend      │◄──►│   (Pickle/Joblib)│◄──►│   (CSV/Excel)   │
    │                 │    │                  │    │                 │
    │ - User Interface│    │ - CatBoost       │    │ - Reviews Data  │
    │ - Visualizations│    │ - Random Forest  │    │ - Preprocessed  │
    │ - Interactions  │    │ - XGBoost        │    │ - Metadata      │
    └─────────────────┘    └──────────────────┘    └─────────────────┘
    ```
    """)
    
    # Technology stack
    st.markdown("#### 🛠️ Technology Stack")
    
    tech_stack = {
        "Frontend": {
            "Streamlit": "1.28.0 - Web application framework",
            "Plotly": "5.15.0 - Interactive visualizations", 
            "Matplotlib": "3.7.0 - Static plots and charts"
        },
        "Machine Learning": {
            "scikit-learn": "1.3.0 - Core ML algorithms",
            "XGBoost": "1.7.6 - Gradient boosting",
            "CatBoost": "1.2.0 - Categorical boosting",
            "LightGBM": "4.0.0 - Light gradient boosting"
        },
        "Data Processing": {
            "Pandas": "2.0.3 - Data manipulation",
            "NumPy": "1.24.3 - Numerical computing",
            "Underthesea": "6.7.0 - Vietnamese NLP"
        },
        "Deployment": {
            "Docker": "24.0.0 - Containerization",
            "Streamlit Cloud": "Community - Hosting",
            "Git": "2.41.0 - Version control"
        }
    }
    
    for category, tools in tech_stack.items():
        st.markdown(f"**{category}:**")
        for tool, description in tools.items():
            st.write(f"- {tool}: {description}")
        st.write("")
    
    # File structure
    st.markdown("#### 📁 Project Structure")
    
    st.markdown("""
    ```
    it_viec_project1/
    ├── app.py                      # Main Streamlit application
    ├── requirements.txt            # Python dependencies
    ├── README.md                   # Project documentation
    │
    ├── pages/                      # Streamlit pages
    │   ├── business_overview.py    # Business context and objectives
    │   ├── sentiment_analysis.py   # Sentiment prediction interface
    │   ├── clustering.py           # Clustering analysis tools
    │   ├── data_exploration.py     # Data visualization and EDA
    │   ├── model_training.py       # Model performance and training
    │   └── documentation.py        # This documentation page
    │
    ├── models/                     # Trained ML models
    │   ├── catboost_classifier.pkl
    │   ├── vectorizer.pkl
    │   └── preprocessor.pkl
    │
    ├── data/                       # Dataset files
    │   ├── reviews.csv
    │   └── final_data.xlsx
    │
    ├── utils/                      # Utility functions
    │   ├── preprocessing.py
    │   ├── model_utils.py
    │   └── visualization.py
    │
    └── notebooks/                  # Jupyter notebooks
        ├── sentiment_analysis.ipynb
        └── clustering_analysis.ipynb
    ```
    """)
    
    # Installation guide
    st.markdown("#### ⚙️ Installation & Setup")
    
    st.markdown("""
    **Prerequisites:**
    - Python 3.8 or higher
    - Git (for cloning repository)
    - 8GB+ RAM recommended
    
    **Installation Steps:**
    
    1. **Clone the Repository:**
    ```bash
    git clone https://github.com/your-repo/it_viec_project1.git
    cd it_viec_project1
    ```
    
    2. **Create Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate
    ```
    
    3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
    4. **Download Models:**
    ```bash
    # Ensure model files are in the models/ directory
    # Contact team for access to trained models
    ```
    
    5. **Run Application:**
    ```bash
    streamlit run app.py
    ```
    
    6. **Access Application:**
    - Open browser to `http://localhost:8501`
    - Navigate through different sections using the sidebar
    """)
    
    # Configuration
    st.markdown("#### ⚙️ Configuration")
    
    st.markdown("""
    **Environment Variables:**
    ```bash
    # Optional configuration
    STREAMLIT_THEME=light           # UI theme
    STREAMLIT_SERVER_PORT=8501      # Server port
    MODEL_PATH=./models/            # Model directory
    DATA_PATH=./data/               # Data directory
    ```
    
    **Streamlit Configuration (.streamlit/config.toml):**
    ```toml
    [theme]
    primaryColor = "#667eea"
    backgroundColor = "#ffffff"
    secondaryBackgroundColor = "#f0f2f6"
    textColor = "#262730"
    
    [server]
    enableCORS = false
    enableXsrfProtection = false
    maxUploadSize = 200
    ```
    """)

with tab3:
    st.markdown("### 📓 Jupyter Notebooks")
    
    st.markdown("""
    Our project includes comprehensive Jupyter notebooks that demonstrate the complete 
    machine learning pipeline from data exploration to model deployment.
    """)
    
    # Notebook descriptions
    notebooks = [
        {
            "Name": "🎯 Project 1 - Exe 1 - Sentiment Analysis.ipynb",
            "Description": "Complete sentiment analysis pipeline including data loading, preprocessing, model training, and evaluation",
            "Location": "../it_viec_sentiment_analysis/Project1/",
            "Key Sections": [
                "Data loading and exploration",
                "Vietnamese text preprocessing with Underthesea",
                "Feature engineering with TF-IDF",
                "Multiple ML model training and comparison",
                "Model evaluation and validation",
                "Performance visualization"
            ]
        },
        {
            "Name": "🔍 bai2_clustering_main.ipynb", 
            "Description": "Information clustering analysis using K-Means and LDA topic modeling",
            "Location": "../it_viec_sentiment_analysis/Project1/",
            "Key Sections": [
                "Data preprocessing for clustering",
                "K-Means clustering implementation",
                "LDA topic modeling",
                "Cluster evaluation and interpretation",
                "Visualization of clustering results",
                "Business insights extraction"
            ]
        },
        {
            "Name": "📊 Content Based Suggestion.ipynb",
            "Description": "Content-based recommendation system using company reviews",
            "Location": "../it_viec/",
            "Key Sections": [
                "Company profile analysis",
                "Content similarity computation", 
                "Recommendation algorithm implementation",
                "Evaluation metrics",
                "User interface components"
            ]
        }
    ]
    
    for notebook in notebooks:
        with st.expander(f"{notebook['Name']}"):
            st.write(f"**Description:** {notebook['Description']}")
            st.write(f"**Location:** `{notebook['Location']}`")
            st.write("**Key Sections:**")
            for section in notebook['Key Sections']:
                st.write(f"- {section}")
    
    # How to run notebooks
    st.markdown("#### 🚀 Running the Notebooks")
    
    st.markdown("""
    **Prerequisites:**
    ```bash
    pip install jupyter pandas numpy scikit-learn plotly matplotlib seaborn
    pip install underthesea gensim wordcloud openpyxl
    ```
    
    **Launch Jupyter:**
    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```
    
    **Notebook Execution Order:**
    1. Start with sentiment analysis notebook for data understanding
    2. Run clustering notebook for pattern discovery
    3. Explore recommendation notebook for advanced techniques
    
    **Tips for Best Results:**
    - Run cells sequentially from top to bottom
    - Ensure all required data files are accessible
    - Install missing packages as needed
    - Clear output and restart kernel if encountering issues
    """)
    
    # Notebook outputs
    st.markdown("#### 📊 Expected Outputs")
    
    st.markdown("""
    **Sentiment Analysis Notebook:**
    - Model performance comparison tables
    - Confusion matrices and classification reports
    - ROC curves and precision-recall curves
    - Feature importance visualizations
    - Prediction examples and error analysis
    
    **Clustering Notebook:**
    - Cluster visualization plots (PCA projection)
    - Word clouds for each cluster
    - Topic distribution charts
    - Cluster evaluation metrics
    - Business insight summaries
    
    **Recommendation Notebook:**
    - Company similarity matrices
    - Recommendation accuracy metrics
    - Interactive recommendation examples
    - Content analysis visualizations
    """)

with tab4:
    st.markdown("### 🌐 API Reference")
    
    # Core functions
    st.markdown("#### 🔧 Core Functions")
    
    api_functions = [
        {
            "Function": "predict_sentiment(text, models, model_name)",
            "Description": "Predict sentiment for given text using specified model",
            "Parameters": [
                "text (str): Input text to analyze",
                "models (dict): Dictionary of loaded models",
                "model_name (str): Name of model to use"
            ],
            "Returns": "tuple: (prediction, confidence_score)",
            "Example": """
            prediction, confidence = predict_sentiment(
                "Công ty rất tốt, môi trường làm việc tuyệt vời",
                models,
                "CatBoost"
            )
            # Returns: (2, 0.95) # 2=positive, 95% confidence
            """
        },
        {
            "Function": "perform_kmeans_clustering(texts, n_clusters)",
            "Description": "Perform K-means clustering on text data",
            "Parameters": [
                "texts (list): List of text documents",
                "n_clusters (int): Number of clusters"
            ],
            "Returns": "tuple: (clusters, X_pca, vectorizer, model)",
            "Example": """
            clusters, X_pca, vectorizer, model = perform_kmeans_clustering(
                review_texts,
                n_clusters=5
            )
            # Returns cluster assignments and visualization data
            """
        },
        {
            "Function": "perform_lda_topic_modeling(texts, n_topics)",
            "Description": "Perform LDA topic modeling on text collection",
            "Parameters": [
                "texts (list): List of text documents",
                "n_topics (int): Number of topics to discover"
            ],
            "Returns": "tuple: (topics, lda_model, vectorizer, lda_matrix)",
            "Example": """
            topics, lda_model, vectorizer, matrix = perform_lda_topic_modeling(
                review_texts,
                n_topics=3
            )
            # Returns topic assignments and model components
            """
        }
    ]
    
    for func in api_functions:
        with st.expander(f"📎 {func['Function']}"):
            st.write(f"**Description:** {func['Description']}")
            st.write("**Parameters:**")
            for param in func['Parameters']:
                st.write(f"- {param}")
            st.write(f"**Returns:** {func['Returns']}")
            st.code(func['Example'], language='python')
    
    # Model loading
    st.markdown("#### 🤖 Model Loading")
    
    st.markdown("""
    **Load Pre-trained Models:**
    ```python
    import joblib
    import pickle
    
    # Load sentiment analysis models
    def load_models():
        models = {}
        
        # Load classifier
        models['catboost'] = joblib.load('models/catboost_classifier.pkl')
        models['random_forest'] = joblib.load('models/randomforest_classifier.pkl')
        
        # Load preprocessing components
        models['vectorizer'] = joblib.load('models/vectorizer.pkl')
        models['preprocessor'] = joblib.load('models/preprocessor.pkl')
        
        return models
    
    # Usage
    models = load_models()
    ```
    """)
    
    # Data processing
    st.markdown("#### 📊 Data Processing")
    
    st.markdown("""
    **Text Preprocessing:**
    ```python
    import re
    from underthesea import word_tokenize
    
    def preprocess_text(text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters (keep Vietnamese)
        text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize Vietnamese text
        tokens = word_tokenize(text)
        
        return ' '.join(tokens)
    
    # Usage
    clean_text = preprocess_text("Công ty rất tốt!!!")
    ```
    
    **Data Loading:**
    ```python
    import pandas as pd
    
    def load_data(file_path):
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        
        # Remove missing values
        df = df.dropna(subset=['processed'])
        
        return df
    
    # Usage
    df = load_data('data/reviews.csv')
    ```
    """)
    
    # Error handling
    st.markdown("#### ⚠️ Error Handling")
    
    st.markdown("""
    **Common Error Patterns:**
    ```python
    try:
        # Model prediction
        prediction, confidence = predict_sentiment(text, models, model_name)
        
        if prediction is None:
            st.error("Prediction failed - check model loading")
            
    except KeyError:
        st.error("Model not found - check model name")
        
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    
    # Data loading errors
    try:
        df = pd.read_csv(file_path)
        
    except FileNotFoundError:
        st.error(f"Data file not found: {file_path}")
        
    except pd.errors.EmptyDataError:
        st.error("Data file is empty")
    ```
    """)

with tab5:
    st.markdown("### ❓ Frequently Asked Questions")
    
    # General questions
    st.markdown("#### 🌟 General Questions")
    
    general_faqs = [
        {
            "Question": "What is the ITViec Analytics Platform?",
            "Answer": """
            The ITViec Analytics Platform is a comprehensive machine learning solution for analyzing 
            employee and candidate reviews. It provides sentiment analysis to classify feedback as 
            positive, negative, or neutral, and clustering analysis to identify patterns and themes 
            in reviews. The platform helps companies understand their reputation and identify 
            improvement areas.
            """
        },
        {
            "Question": "What data does the platform analyze?",
            "Answer": """
            The platform analyzes over 30,000 employee and candidate reviews from ITViec, covering 
            multiple companies across various industries. The data includes detailed text reviews, 
            ratings across different categories (salary, culture, management, etc.), and company 
            information. All data is processed to ensure privacy and anonymity.
            """
        },
        {
            "Question": "How accurate are the predictions?",
            "Answer": """
            Our best-performing model (CatBoost) achieves 95.2% accuracy on sentiment classification. 
            The models are trained on Vietnamese text data and validated using cross-validation 
            techniques. Performance varies slightly based on text length and complexity, with longer, 
            more detailed reviews typically yielding higher accuracy.
            """
        }
    ]
    
    for faq in general_faqs:
        with st.expander(f"❓ {faq['Question']}"):
            st.write(faq['Answer'])
    
    # Technical questions
    st.markdown("#### 🔧 Technical Questions")
    
    technical_faqs = [
        {
            "Question": "Which machine learning models are used?",
            "Answer": """
            We use multiple state-of-the-art machine learning algorithms:
            - **CatBoost**: Best performing model (95.2% accuracy)
            - **Random Forest**: Ensemble method (94.3% accuracy) 
            - **XGBoost**: Gradient boosting (92.1% accuracy)
            - **LightGBM**: Fast gradient boosting (93.4% accuracy)
            - **Logistic Regression**: Linear baseline (89.2% accuracy)
            
            Models are trained using TF-IDF features and optimized hyperparameters.
            """
        },
        {
            "Question": "How is Vietnamese text processed?",
            "Answer": """
            Vietnamese text processing uses the Underthesea library, which provides:
            - **Word segmentation**: Proper tokenization of Vietnamese words
            - **POS tagging**: Part-of-speech identification
            - **Text normalization**: Cleaning and standardization
            - **Special character handling**: Preservation of Vietnamese diacritics
            
            The preprocessing pipeline ensures optimal feature extraction for ML models.
            """
        },
        {
            "Question": "What clustering algorithms are implemented?",
            "Answer": """
            Two main clustering approaches are available:
            
            **K-Means Clustering:**
            - Groups reviews based on content similarity
            - Fixed number of clusters (user-defined)
            - Fast and efficient for large datasets
            - Uses TF-IDF vectorization
            
            **LDA Topic Modeling:**
            - Discovers hidden topics in review collections
            - Probabilistic topic assignments
            - Interpretable word-topic distributions
            - Automatic theme identification
            """
        }
    ]
    
    for faq in technical_faqs:
        with st.expander(f"🔧 {faq['Question']}"):
            st.write(faq['Answer'])
    
    # Usage questions
    st.markdown("#### 💡 Usage Questions")
    
    usage_faqs = [
        {
            "Question": "How do I analyze sentiment for my own text?",
            "Answer": """
            1. Navigate to the **Sentiment Analysis** tab
            2. Enter your text in the input area (Vietnamese or English supported)
            3. Select your preferred model (CatBoost recommended)
            4. Click **Analyze Sentiment**
            5. View results including sentiment classification and confidence score
            
            For batch analysis, upload a CSV file with text in one column.
            """
        },
        {
            "Question": "How do I perform clustering analysis?",
            "Answer": """
            1. Go to the **Information Clustering** tab
            2. Choose clustering method (K-Means or LDA)
            3. Set the number of clusters/topics (3-5 recommended for initial exploration)
            4. Click **Run Clustering Analysis**
            5. Explore results in different sub-tabs:
               - Overview: cluster statistics and visualizations
               - Details: in-depth cluster analysis
               - Word Clouds: visual representation of cluster themes
            """
        },
        {
            "Question": "Can I download the analysis results?",
            "Answer": """
            Yes! Download options are available throughout the platform:
            - **Sentiment Analysis**: Download batch analysis results as CSV
            - **Data Exploration**: Download full dataset and summary statistics
            - **Model Training**: Download performance reports and training logs
            
            All downloads include timestamps and are formatted for further analysis.
            """
        }
    ]
    
    for faq in usage_faqs:
        with st.expander(f"💡 {faq['Question']}"):
            st.write(faq['Answer'])
    
    # Troubleshooting
    st.markdown("#### 🛠️ Troubleshooting")
    
    troubleshooting = [
        {
            "Issue": "Models not loading / 'Models not found' error",
            "Solution": """
            **Possible causes and solutions:**
            1. **Model files missing**: Ensure model files exist in the correct directory
            2. **Incorrect file paths**: Check that paths point to actual model locations
            3. **File corruption**: Re-download model files if available
            4. **Permission issues**: Verify read access to model directory
            
            **Debug steps:**
            ```python
            import os
            model_path = "../it_viec_sentiment_analysis/Project1/best_ml_models"
            print(f"Path exists: {os.path.exists(model_path)}")
            print(f"Files: {os.listdir(model_path) if os.path.exists(model_path) else 'Not found'}")
            ```
            """
        },
        {
            "Issue": "Poor prediction accuracy",
            "Solution": """
            **Improvement strategies:**
            1. **Text quality**: Ensure input text is clear and grammatically correct
            2. **Text length**: Longer texts (50+ words) typically yield better results
            3. **Language**: Vietnamese text processes better than mixed languages
            4. **Model selection**: Try different models for comparison
            5. **Preprocessing**: Check if text needs additional cleaning
            
            **Best practices:**
            - Use complete sentences rather than fragments
            - Avoid excessive special characters or formatting
            - Provide context when possible
            """
        },
        {
            "Issue": "Clustering results seem unclear",
            "Solution": """
            **Optimization approaches:**
            1. **Cluster count**: Try different numbers (3-7 typically work well)
            2. **Text preprocessing**: Ensure consistent text cleaning
            3. **Algorithm choice**: Compare K-Means vs LDA results
            4. **Data quality**: Remove very short or irrelevant texts
            5. **Parameter tuning**: Adjust vectorization parameters
            
            **Interpretation tips:**
            - Look at top keywords for each cluster
            - Examine sample reviews from each cluster
            - Consider business context when labeling clusters
            """
        }
    ]
    
    for item in troubleshooting:
        with st.expander(f"🛠️ {item['Issue']}"):
            st.write(item['Solution'])
    
    # Contact information
    st.markdown("#### 📞 Support & Contact")
    
    st.markdown("""
    **Need additional help?**
    
    📧 **Email Support**: dao.tuan.thinh@student.example.edu, truong.van.le@student.example.edu
    
    📚 **Additional Resources**:
    - Project repository documentation
    - Jupyter notebook examples
    - Model training logs and reports
    
    🔄 **Updates**: This platform is actively maintained. Check back for new features and improvements.
    
    💡 **Feedback**: We welcome suggestions for improvements and new features!
    """)

# Footer with additional resources
st.markdown("---")
st.markdown("## 📚 Additional Resources")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📖 Learning Resources**
    - [Streamlit Documentation](https://docs.streamlit.io)
    - [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
    - [Vietnamese NLP with Underthesea](https://underthesea.readthedocs.io)
    """)

with col2:
    st.markdown("""
    **🛠️ Development Tools**
    - [Jupyter Notebook](https://jupyter.org)
    - [Plotly Documentation](https://plotly.com/python)
    - [Pandas Documentation](https://pandas.pydata.org/docs)
    """)

with col3:
    st.markdown("""
    **📊 Data Science**
    - [Sentiment Analysis Guide](https://en.wikipedia.org/wiki/Sentiment_analysis)
    - [Clustering Algorithms](https://scikit-learn.org/stable/modules/clustering.html)
    - [Text Mining Techniques](https://en.wikipedia.org/wiki/Text_mining)
    """)

# Version information
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    ITViec Analytics Platform v1.0 | Last Updated: {datetime.now().strftime('%B %Y')} | 
    Built with ❤️ by Đào Tuấn Thịnh & Trương Văn Lê
</div>
""", unsafe_allow_html=True)
