import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the VietnamesePreprocessor class to make it available for model loading
from utils.vietnamese_preprocessor import VietnamesePreprocessor

# Configure page
st.set_page_config(
    page_title="🎯 ITViec Analytics Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.itviec.com',
        'Report a bug': None,
        'About': """
        # ITViec Sentiment Analysis & Clustering Platform
        
        This application provides comprehensive analysis tools for:
        - **Sentiment Analysis**: Analyze employee/candidate feedback sentiment
        - **Information Clustering**: Group reviews to identify patterns
        
        Built for ITViec to enhance company reputation and workplace improvements.
        """
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .sidebar-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - Team Information
with st.sidebar:
    st.markdown("""
    <div class="sidebar-info">
        <h4 style="margin-top: 0; color: #28a745;">👥 Development Team</h4>
        <p><strong>🎓 Students:</strong></p>
        <ul style="margin-left: 1rem;">
            <li>Đào Tuấn Thịnh</li>
            <li>Trương Văn Lê</li>
        </ul>
        <p><strong>👨‍🏫 Supervisor:</strong><br>Khuất Thị Phương</p>
        <hr>
        <p style="font-size: 0.8rem; color: #666;">
            🏢 <strong>ITViec Analytics Platform</strong><br>
            Advanced Sentiment Analysis & Clustering Solutions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    st.markdown("### 🧭 Navigation")
    page = st.selectbox(
        "Choose a section:",
        [
            "🏠 Home",
            "📊 Business Overview", 
            "🎯 Sentiment Analysis",
            "🔍 Information Clustering",
            "🏢 Company Analysis",
            "📈 Data Exploration",
            "🔬 Model Training",
            "📚 Documentation"
        ]
    )

# Main content based on selection
if page == "🏠 Home":
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 ITViec Analytics Platform</h1>
        <p style="font-size: 1.2rem; margin-bottom: 0;">
            Advanced Sentiment Analysis & Information Clustering for Better Workplace Insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome message
    st.markdown("## 👋 Welcome to ITViec Analytics Platform")
    
    st.markdown("""
    This comprehensive platform provides cutting-edge analytics solutions designed specifically for ITViec 
    and partner companies to understand employee and candidate sentiment, enabling data-driven decisions 
    for workplace improvements and enhanced company reputation.
    """)
    
    # Key Features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Sentiment Analysis</h3>
            <p>Analyze employee and candidate feedback with advanced machine learning models to classify sentiment as positive, negative, or neutral.</p>
            <ul>
                <li>Real-time sentiment prediction</li>
                <li>Multiple ML models comparison</li>
                <li>Vietnamese text processing</li>
                <li>Interactive visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Information Clustering</h3>
            <p>Group reviews and feedback to identify patterns and themes, helping companies understand evaluation clusters and improvement areas.</p>
            <ul>
                <li>Advanced clustering algorithms</li>
                <li>Topic modeling with LDA</li>
                <li>Pattern recognition</li>
                <li>Actionable insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🏢 Company Analysis</h3>
            <p>Deep dive into individual company performance with clustering analysis, keyword extraction, and competitive benchmarking.</p>
            <ul>
                <li>Company-specific clustering</li>
                <li>Word cloud visualization</li>
                <li>Performance ranking</li>
                <li>Competitive comparison</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Stats
    st.markdown("## 📊 Platform Capabilities")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>10+</h3>
            <p>ML Models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>95%+</h3>
            <p>Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>30K+</h3>
            <p>Reviews Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Real-time</h3>
            <p>Processing</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting Started
    st.markdown("## 🚀 Getting Started")
    
    st.markdown("""
    1. **📊 Business Overview**: Learn about our objectives and requirements
    2. **🎯 Sentiment Analysis**: Try our sentiment prediction models
    3. **🔍 Information Clustering**: Explore review clustering and topic analysis
    4. **🏢 Company Analysis**: Analyze individual companies with clustering & benchmarking
    5. **📈 Data Exploration**: Examine the underlying data and patterns
    6. **🔬 Model Training**: See how our models are trained and validated
    7. **📚 Documentation**: Access technical documentation and notebooks
    """)
    
    # Quick Actions
    st.markdown("## ⚡ Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🎯 Try Sentiment Analysis", use_container_width=True):
            st.switch_page("pages/sentiment_analysis.py")
    
    with col2:
        if st.button("🔍 Explore Clustering", use_container_width=True):
            st.switch_page("pages/clustering.py")
    
    with col3:
        if st.button("🏢 Company Analysis", use_container_width=True):
            st.switch_page("pages/company_analysis.py")
    
    with col4:
        if st.button("📊 Business Overview", use_container_width=True):
            st.switch_page("pages/business_overview.py")

elif page == "📊 Business Overview":
    exec(open("pages/business_overview.py").read())
elif page == "🎯 Sentiment Analysis":
    exec(open("pages/sentiment_analysis.py").read())
elif page == "🔍 Information Clustering":
    exec(open("pages/clustering.py").read())
elif page == "🏢 Company Analysis":
    exec(open("pages/company_analysis.py").read())
elif page == "📈 Data Exploration":
    exec(open("pages/data_exploration.py").read())
elif page == "🔬 Model Training":
    exec(open("pages/model_training.py").read())
elif page == "📚 Documentation":
    exec(open("pages/documentation.py").read())