import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page header
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
            padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
    <h1>📊 Business Overview</h1>
    <p style="font-size: 1.2rem; margin-bottom: 0;">Understanding the Business Context and Objectives</p>
</div>
""", unsafe_allow_html=True)

# Business Objectives
st.markdown("## 🎯 Business Objectives")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 💭 Sentiment Analysis Objective
    
    **Goal**: Develop a predictive model to quickly analyze feedback from employees 
    or candidates about companies, classifying sentiments as:
    - ✅ **Positive**: Favorable reviews and experiences
    - ❌ **Negative**: Critical feedback requiring attention  
    - ⚖️ **Neutral**: Balanced or factual observations
    
    **Business Value**:
    - 📈 Improve company reputation management
    - 😊 Enhance employee satisfaction tracking
    - ⚡ Enable real-time sentiment monitoring
    - 📊 Provide data-driven insights for HR decisions
    """)

with col2:
    st.markdown("""
    ### 🔍 Information Clustering Objective
    
    **Goal**: Build clustering models to group employee/candidate reviews and 
    identify patterns in feedback themes such as:
    - 💰 Salary & Benefits
    - 📚 Training & Learning
    - 👥 Management & Culture
    - 🏢 Office Environment
    
    **Business Value**:
    - 🎯 Identify improvement areas by cluster analysis
    - 📈 Understand evaluation patterns across companies
    - 🔄 Enable targeted workplace improvements
    - 💡 Generate actionable insights for better work environments
    """)

# Key Requirements
st.markdown("## 📋 Project Requirements")

tab1, tab2, tab3 = st.tabs(["🔧 Technical Requirements", "📊 Data Requirements", "🎯 Performance Requirements"])

with tab1:
    st.markdown("""
    ### Technical Specifications
    
    **Machine Learning Models**:
    - 🤖 Multiple classification algorithms (Random Forest, XGBoost, CatBoost, etc.)
    - 📊 Clustering algorithms (K-Means, LDA Topic Modeling)
    - 🔤 Vietnamese text preprocessing with Underthesea
    - ⚡ Real-time prediction capabilities
    
    **Technology Stack**:
    - 🐍 Python ecosystem (scikit-learn, pandas, numpy)
    - 🎨 Streamlit for interactive web application
    - 📈 Plotly for dynamic visualizations
    - 💾 Pickle/Joblib for model serialization
    
    **Features**:
    - 🌐 Bilingual support (Vietnamese & English)
    - 📱 Responsive web interface
    - 📊 Interactive data exploration
    - 🔄 Model comparison and evaluation
    """)

with tab2:
    st.markdown("""
    ### Data Specifications
    
    **Dataset Information**:
    - 📦 **Source**: ITViec employee/candidate reviews
    - 📊 **Size**: 30,000+ review records
    - 🔤 **Language**: Primarily Vietnamese with some English
    - 📅 **Time Range**: Multi-year historical data
    
    **Key Features**:
    - 🏢 **Company Name**: Organization being reviewed
    - ⭐ **Ratings**: Numerical scores across multiple dimensions
    - 📝 **Text Reviews**: Detailed feedback in multiple categories
    - 📊 **Structured Metrics**: Salary, culture, management ratings
    
    **Data Quality**:
    - ✅ Cleaned and preprocessed text data
    - 🔍 Labeled sentiment classifications
    - 📊 Balanced representation across companies
    - 🎯 Validated clustering ground truth
    """)

with tab3:
    st.markdown("""
    ### Performance Standards
    
    **Sentiment Analysis Targets**:
    - 🎯 **Accuracy**: > 90% on test dataset
    - ⚡ **Speed**: < 100ms prediction time
    - 📊 **Precision/Recall**: Balanced across all sentiment classes
    - 🔄 **Robustness**: Consistent performance across company types
    
    **Clustering Performance**:
    - 📊 **Coherence Score**: > 0.4 for topic models
    - 🎯 **Silhouette Score**: > 0.3 for K-means clustering
    - 📈 **Interpretability**: Clear, actionable cluster themes
    - 🔍 **Stability**: Consistent clusters across runs
    
    **System Performance**:
    - 🚀 **Load Time**: < 3 seconds for app initialization
    - 💾 **Memory Usage**: < 2GB RAM for full application
    - 🌐 **Scalability**: Support for 100+ concurrent users
    - 📱 **Responsiveness**: Mobile-friendly interface
    """)

# Success Metrics
st.markdown("## 📈 Success Metrics & KPIs")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1rem; border-radius: 8px; text-align: center; color: white;">
        <h3>95%+</h3>
        <p>Model Accuracy Target</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                padding: 1rem; border-radius: 8px; text-align: center; color: white;">
        <h3>30K+</h3>
        <p>Reviews Processed</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                padding: 1rem; border-radius: 8px; text-align: center; color: white;">
        <h3>10+</h3>
        <p>ML Models Compared</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                padding: 1rem; border-radius: 8px; text-align: center; color: white;">
        <h3>Real-time</h3>
        <p>Prediction Speed</p>
    </div>
    """, unsafe_allow_html=True)

# Business Impact
st.markdown("## 💼 Expected Business Impact")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🏢 For ITViec Platform
    
    **Enhanced Services**:
    - 📊 Provide sentiment analytics as a premium service
    - 🎯 Help companies understand their reputation
    - 📈 Increase platform value proposition
    - 🤝 Strengthen relationships with partner companies
    
    **Competitive Advantages**:
    - 🚀 First mover in Vietnamese job market analytics
    - 🔬 Advanced AI/ML capabilities
    - 📊 Data-driven insights for clients
    - 💡 Innovation leadership in HR tech
    """)

with col2:
    st.markdown("""
    ### 🏭 For Partner Companies
    
    **Operational Benefits**:
    - 😊 Improved employee satisfaction
    - 📈 Better talent retention rates
    - 🎯 Targeted workplace improvements
    - 📊 Data-driven HR strategies
    
    **Strategic Advantages**:
    - 🌟 Enhanced employer branding
    - 🔍 Competitive intelligence
    - 📈 Measurable culture improvements
    - 💼 Reduced recruitment costs
    """)

# Implementation Roadmap
st.markdown("## 🗺️ Implementation Roadmap")

# Create a simple timeline
timeline_data = {
    'Phase': ['Data Collection', 'Model Development', 'Validation', 'Deployment', 'Monitoring'],
    'Duration': ['2 weeks', '4 weeks', '2 weeks', '1 week', 'Ongoing'],
    'Key Activities': [
        'Data gathering, cleaning, preprocessing',
        'Model training, hyperparameter tuning, comparison',
        'Performance testing, user acceptance testing',
        'Production deployment, system integration',
        'Performance monitoring, model updates'
    ],
    'Status': ['✅ Complete', '✅ Complete', '✅ Complete', '🟡 In Progress', '🔄 Continuous']
}

timeline_df = pd.DataFrame(timeline_data)

st.dataframe(
    timeline_df,
    use_container_width=True,
    hide_index=True
)

# Risk Assessment
st.markdown("## ⚠️ Risk Assessment & Mitigation")

risk_col1, risk_col2 = st.columns(2)

with risk_col1:
    st.markdown("""
    ### 🔴 Identified Risks
    
    **Technical Risks**:
    - 🔤 Vietnamese text processing complexity
    - 📊 Model overfitting on limited data
    - ⚡ Performance degradation at scale
    - 🔄 Concept drift over time
    
    **Business Risks**:
    - 📈 Low user adoption rates
    - 🎯 Misaligned expectations
    - 💰 Budget constraints
    - ⏰ Timeline delays
    """)

with risk_col2:
    st.markdown("""
    ### 🟢 Mitigation Strategies
    
    **Technical Mitigations**:
    - 🛠️ Use proven NLP libraries (Underthesea)
    - 📊 Implement cross-validation and regularization
    - ⚡ Optimize algorithms and use caching
    - 🔄 Monitor model performance continuously
    
    **Business Mitigations**:
    - 👥 Regular stakeholder communication
    - 🎯 Clear requirement documentation
    - 💰 Phased implementation approach
    - ⏰ Agile development methodology
    """)

# Call to Action
st.markdown("## 🚀 Next Steps")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎯 Try Sentiment Analysis", use_container_width=True):
        st.info("Navigate to Sentiment Analysis section to test our models!")

with col2:
    if st.button("🔍 Explore Clustering", use_container_width=True):
        st.info("Check out our clustering models and topic analysis!")

with col3:
    if st.button("📊 View Data Exploration", use_container_width=True):
        st.info("Dive into the data insights and patterns!")
