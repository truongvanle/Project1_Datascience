# 🎯 ITViec Analytics Platform

A comprehensive Streamlit application for sentiment analysis and information clustering of employee and candidate reviews from ITViec.

## 🌟 Overview

The ITViec Analytics Platform provides advanced machine learning solutions for:

- **🎯 Sentiment Analysis**: Classify employee/candidate feedback as positive, negative, or neutral
- **🔍 Information Clustering**: Group reviews to identify patterns and improvement areas
- **🏢 Company Analysis**: Deep dive into individual company performance with clustering & benchmarking
- **📊 Data Exploration**: Interactive analysis of review datasets
- **🔬 Model Training**: View ML model performance and training details

## ✨ Features

### 🎯 Sentiment Analysis
- Real-time sentiment prediction for Vietnamese and English text
- Multiple ML models comparison (CatBoost, Random Forest, XGBoost, etc.)
- Batch processing for large datasets
- Confidence scoring and model comparison
- Interactive visualizations

### 🔍 Information Clustering
- K-Means clustering for content-based grouping
- LDA topic modeling for theme discovery
- Word cloud visualizations
- Cluster analysis and interpretation
- Business insights extraction

### 🏢 Company Analysis
- Company-specific clustering analysis
- Performance benchmarking against other companies
- Word cloud visualization of company-specific themes
- Overall rating and score comparisons
- Temporal analysis of company reviews
- Recent reviews showcase
- Cluster-based insights for individual companies

### 📊 Data Exploration
- Comprehensive dataset overview
- Company and rating analysis
- Temporal trend analysis
- Text analytics and statistics
- Interactive visualizations

### 🔬 Model Training
- Model performance comparison
- Hyperparameter optimization details
- Training process visualization
- Cross-validation results
- Model deployment information

## 🛠️ Technology Stack

- **Frontend**: Streamlit 1.28.0+
- **ML Libraries**: scikit-learn, XGBoost, CatBoost, LightGBM
- **NLP**: Underthesea (Vietnamese), Gensim
- **Visualization**: Plotly, Matplotlib, WordCloud
- **Data Processing**: Pandas, NumPy

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM recommended

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd it_viec_project1
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
streamlit run app.py
```

5. **Access the application:**
Open your browser to `http://localhost:8501`

## 📁 Project Structure

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
│   └── documentation.py        # Technical documentation
│
├── models/                     # Trained ML models
│   ├── catboost_classifier.pkl # Best performing model
│   ├── vectorizer.pkl          # TF-IDF vectorizer
│   └── preprocessor.pkl        # Text preprocessor
│
├── data/                       # Dataset files
│   ├── reviews.csv             # Raw reviews data
│   └── final_data.xlsx         # Preprocessed data
│
├── utils/                      # Utility functions
│   ├── preprocessing.py        # Text preprocessing utilities
│   ├── model_utils.py         # ML model utilities
│   └── visualization.py       # Plotting and visualization
│
└── notebooks/                  # Jupyter notebooks
    ├── Project 1 - Exe 1 - Sentiment Analysis.ipynb
    └── bai2_clustering_main.ipynb
```

## 🎯 Business Objectives

### Sentiment Analysis Goal
Develop predictive models to quickly analyze employee/candidate feedback, enabling:
- Real-time sentiment monitoring
- Company reputation management
- Data-driven HR decisions
- Employee satisfaction tracking

### Information Clustering Goal
Build clustering models to identify review patterns and themes:
- Targeted workplace improvements
- Evaluation pattern recognition
- Actionable business insights
- Competitive intelligence

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| CatBoost | 95.2% | 95.1% | 95.2% | 95.1% |
| Random Forest | 94.3% | 94.1% | 94.3% | 94.2% |
| XGBoost | 92.1% | 91.9% | 92.1% | 92.0% |
| LightGBM | 93.4% | 93.2% | 93.4% | 93.3% |
| Logistic Regression | 89.2% | 89.0% | 89.2% | 89.1% |

## 🔧 Usage Examples

### Sentiment Analysis
```python
# Single text prediction
prediction, confidence = predict_sentiment(
    "Công ty rất tốt, môi trường làm việc tuyệt vời",
    models,
    "CatBoost"
)
```

### Clustering Analysis
```python
# K-means clustering
clusters, X_pca, vectorizer, model = perform_kmeans_clustering(
    review_texts,
    n_clusters=5
)
```

## 📈 Key Features

- **🌐 Vietnamese NLP**: Advanced processing with Underthesea
- **⚡ Real-time Predictions**: Fast sentiment analysis (< 100ms)
- **📊 Interactive Visualizations**: Dynamic charts and plots
- **🔄 Model Comparison**: Multiple algorithms side-by-side
- **📱 Responsive Design**: Works on desktop and mobile
- **💾 Export Capabilities**: Download results and reports

## 🛡️ Data Privacy

- All personal information is anonymized
- No individual identification possible
- Aggregate analysis only
- GDPR compliant processing

## 👥 Team

**Students:**
- Đào Tuấn Thịnh
- Trương Văn Lê

**Supervisor:**
- Khuất Thị Phương

## 📞 Support

For questions or issues:
- Check the Documentation section in the app
- Review the Jupyter notebooks for technical details
- Contact the development team

## 🔄 Updates

This platform is actively maintained with regular updates for:
- Model performance improvements
- New features and visualizations
- Bug fixes and optimizations
- Data pipeline enhancements

## 📜 License

This project is developed for educational purposes as part of an academic program.

---

*Built with ❤️ for ITViec and the Vietnamese tech community*