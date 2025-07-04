import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Page header
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
            padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
    <h1>🔍 Information Clustering</h1>
    <p style="font-size: 1.2rem; margin-bottom: 0;">Discover Patterns in Employee & Candidate Reviews</p>
</div>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_clustering_data():
    """Load data for clustering analysis"""
    # Try local data first
    local_data_path = "data/final_data.xlsx"
    external_data_path = "Project1/final_data.xlsx"
    
    for data_path in [local_data_path, external_data_path]:
        if os.path.exists(data_path):
            try:
                df = pd.read_excel(data_path)
                # Check for processed text column
                if 'processed_review' in df.columns:
                    # Remove rows with missing processed text
                    df = df.dropna(subset=['processed_review'])
                    return df
                elif 'processed' in df.columns:
                    # Fallback to 'processed' column name
                    df = df.dropna(subset=['processed'])
                    return df
                else:
                    # If no processed column, we'll need to create one
                    st.warning("No preprocessed text column found. Using raw text.")
                    # Combine text columns for processing
                    df['processed_review'] = df.apply(lambda row: 
                        ' '.join([str(row.get('Title', '')), 
                                str(row.get('What I liked', '')), 
                                str(row.get('Suggestions for improvement', ''))]), axis=1)
                    df = df.dropna(subset=['processed_review'])
                    return df
            except Exception as e:
                st.error(f"Error loading data from {data_path}: {e}")
                continue
    
    st.error("No suitable data file found for clustering analysis")
    return None

# Text preprocessing
def preprocess_for_clustering(text):
    """Preprocess text for clustering"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Clustering functions
@st.cache_resource
def perform_kmeans_clustering(texts, n_clusters=5):
    """Perform K-means clustering on texts"""
    try:
        # Vectorize texts
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=None, max_df=0.95, min_df=2)
        X = vectorizer.fit_transform(texts)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # PCA for visualization
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X.toarray())
        
        return clusters, X_pca, vectorizer, kmeans
    except Exception as e:
        st.error(f"Clustering error: {e}")
        return None, None, None, None

@st.cache_resource
def perform_lda_topic_modeling(texts, n_topics=5):
    """Perform LDA topic modeling"""
    try:
        # Vectorize texts
        vectorizer = CountVectorizer(max_features=1000, stop_words=None, max_df=0.95, min_df=2)
        doc_term_matrix = vectorizer.fit_transform(texts)
        
        # LDA model
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=100)
        lda_matrix = lda.fit_transform(doc_term_matrix)
        
        # Get dominant topics
        dominant_topics = np.argmax(lda_matrix, axis=1)
        
        return dominant_topics, lda, vectorizer, lda_matrix
    except Exception as e:
        st.error(f"LDA error: {e}")
        return None, None, None, None

def get_top_words_per_cluster(texts, clusters, vectorizer, n_words=10):
    """Get top words for each cluster"""
    cluster_words = {}
    
    for cluster_id in np.unique(clusters):
        cluster_texts = [texts[i] for i in range(len(texts)) if clusters[i] == cluster_id]
        cluster_text = ' '.join(cluster_texts)
        
        # Get TF-IDF scores
        tfidf_matrix = vectorizer.transform([cluster_text])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top words
        scores = tfidf_matrix.toarray()[0]
        top_indices = scores.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_indices if scores[i] > 0]
        
        cluster_words[cluster_id] = top_words
    
    return cluster_words

def get_lda_topics(lda_model, vectorizer, n_words=10):
    """Extract topics from LDA model"""
    topics = {}
    feature_names = vectorizer.get_feature_names_out()
    
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-n_words:][::-1]]
        topics[topic_idx] = top_words
    
    return topics

# Load data
df = load_clustering_data()

if df is not None:
    st.success(f"✅ Loaded {len(df)} reviews for analysis")
    
    # Sidebar options
    st.sidebar.markdown("### 🔧 Clustering Options")
    
    clustering_method = st.sidebar.selectbox(
        "Select Clustering Method:",
        ["K-Means Clustering", "LDA Topic Modeling"]
    )
    
    if clustering_method == "K-Means Clustering":
        n_clusters = st.sidebar.slider("Number of Clusters:", 2, 10, 5)
    else:
        n_clusters = st.sidebar.slider("Number of Topics:", 2, 10, 5)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Clustering Analysis", "🔍 Cluster Details", "☁️ Word Clouds", "📈 Visualizations"])
    
    with tab1:
        st.markdown("### 📊 Clustering Overview")
        
        if st.button("🚀 Run Clustering Analysis", type="primary"):
            with st.spinner(f"Running {clustering_method}..."):
                texts = df['processed_review'].tolist()
                
                if clustering_method == "K-Means Clustering":
                    clusters, X_pca, vectorizer, model = perform_kmeans_clustering(texts, n_clusters)
                    
                    if clusters is not None:
                        # Add results to dataframe
                        df_results = df.copy()
                        df_results['cluster'] = clusters
                        
                        # Cluster statistics
                        st.markdown("#### 📈 Cluster Statistics")
                        
                        cluster_stats = df_results.groupby('cluster').size().reset_index(name='count')
                        cluster_stats['percentage'] = (cluster_stats['count'] / len(df_results) * 100).round(2)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_pie = px.pie(
                                cluster_stats, 
                                values='count', 
                                names='cluster',
                                title="Cluster Size Distribution"
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            fig_bar = px.bar(
                                cluster_stats, 
                                x='cluster', 
                                y='count',
                                title="Reviews per Cluster"
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # PCA Visualization
                        st.markdown("#### 🎯 Cluster Visualization (PCA)")
                        
                        df_viz = pd.DataFrame({
                            'PC1': X_pca[:, 0],
                            'PC2': X_pca[:, 1],
                            'Cluster': clusters.astype(str)
                        })
                        
                        fig_scatter = px.scatter(
                            df_viz, 
                            x='PC1', 
                            y='PC2', 
                            color='Cluster',
                            title="K-Means Clusters (PCA Projection)"
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        # Store results in session state
                        st.session_state['clustering_results'] = df_results
                        st.session_state['cluster_words'] = get_top_words_per_cluster(texts, clusters, vectorizer)
                        st.session_state['clustering_method'] = 'kmeans'
                
                else:  # LDA Topic Modeling
                    topics, lda_model, vectorizer, lda_matrix = perform_lda_topic_modeling(texts, n_clusters)
                    
                    if topics is not None:
                        # Add results to dataframe
                        df_results = df.copy()
                        df_results['topic'] = topics
                        
                        # Topic statistics
                        st.markdown("#### 📈 Topic Statistics")
                        
                        topic_stats = df_results.groupby('topic').size().reset_index(name='count')
                        topic_stats['percentage'] = (topic_stats['count'] / len(df_results) * 100).round(2)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_pie = px.pie(
                                topic_stats, 
                                values='count', 
                                names='topic',
                                title="Topic Distribution"
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            fig_bar = px.bar(
                                topic_stats, 
                                x='topic', 
                                y='count',
                                title="Reviews per Topic"
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # Topic composition heatmap
                        st.markdown("#### 🔥 Topic-Document Probability Matrix")
                        
                        # Sample of probability matrix for visualization
                        sample_size = min(100, len(lda_matrix))
                        sample_indices = np.random.choice(len(lda_matrix), sample_size, replace=False)
                        sample_matrix = lda_matrix[sample_indices]
                        
                        fig_heatmap = px.imshow(
                            sample_matrix.T,
                            labels=dict(x="Document Sample", y="Topic", color="Probability"),
                            title="Topic-Document Probability Matrix (Sample)"
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                        
                        # Store results
                        st.session_state['clustering_results'] = df_results
                        st.session_state['lda_topics'] = get_lda_topics(lda_model, vectorizer)
                        st.session_state['clustering_method'] = 'lda'
    
    with tab2:
        st.markdown("### 🔍 Detailed Cluster Analysis")
        
        if 'clustering_results' in st.session_state:
            df_results = st.session_state['clustering_results']
            method = st.session_state['clustering_method']
            
            if method == 'kmeans':
                cluster_col = 'cluster'
                clusters_list = sorted(df_results[cluster_col].unique())
                cluster_words = st.session_state.get('cluster_words', {})
                
                selected_cluster = st.selectbox("Select Cluster:", clusters_list)
                
                # Cluster details
                cluster_data = df_results[df_results[cluster_col] == selected_cluster]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Cluster Size", len(cluster_data))
                    st.metric("Percentage", f"{len(cluster_data)/len(df_results)*100:.1f}%")
                
                with col2:
                    avg_rating = cluster_data['Rating'].mean() if 'Rating' in cluster_data.columns else 0
                    st.metric("Average Rating", f"{avg_rating:.2f}")
                
                # Top words
                if selected_cluster in cluster_words:
                    st.markdown("#### 🔤 Top Keywords")
                    keywords = cluster_words[selected_cluster][:10]
                    st.write(", ".join(keywords))
                
                # Sample reviews
                st.markdown("#### 📝 Sample Reviews")
                sample_reviews = cluster_data.head(5)
                for idx, row in sample_reviews.iterrows():
                    with st.expander(f"Review {idx + 1}"):
                        if 'Company Name' in row:
                            st.write(f"**Company:** {row['Company Name']}")
                        if 'Title' in row:
                            st.write(f"**Title:** {row['Title']}")
                        if 'What I liked' in row:
                            st.write(f"**Liked:** {row['What I liked']}")
                        if 'Suggestions for improvement' in row:
                            st.write(f"**Suggestions:** {row['Suggestions for improvement']}")
            
            else:  # LDA
                topic_col = 'topic'
                topics_list = sorted(df_results[topic_col].unique())
                lda_topics = st.session_state.get('lda_topics', {})
                
                selected_topic = st.selectbox("Select Topic:", topics_list)
                
                # Topic details
                topic_data = df_results[df_results[topic_col] == selected_topic]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Topic Size", len(topic_data))
                    st.metric("Percentage", f"{len(topic_data)/len(df_results)*100:.1f}%")
                
                with col2:
                    avg_rating = topic_data['Rating'].mean() if 'Rating' in topic_data.columns else 0
                    st.metric("Average Rating", f"{avg_rating:.2f}")
                
                # Top words
                if selected_topic in lda_topics:
                    st.markdown("#### 🔤 Topic Keywords")
                    keywords = lda_topics[selected_topic][:10]
                    st.write(", ".join(keywords))
                
                # Sample reviews
                st.markdown("#### 📝 Sample Reviews")
                sample_reviews = topic_data.head(5)
                for idx, row in sample_reviews.iterrows():
                    with st.expander(f"Review {idx + 1}"):
                        if 'Company Name' in row:
                            st.write(f"**Company:** {row['Company Name']}")
                        if 'Title' in row:
                            st.write(f"**Title:** {row['Title']}")
                        if 'What I liked' in row:
                            st.write(f"**Liked:** {row['What I liked']}")
        else:
            st.info("Run clustering analysis first to see detailed results.")
    
    with tab3:
        st.markdown("### ☁️ Word Clouds")
        
        if 'clustering_results' in st.session_state:
            df_results = st.session_state['clustering_results']
            method = st.session_state['clustering_method']
            
            cluster_col = 'cluster' if method == 'kmeans' else 'topic'
            clusters_list = sorted(df_results[cluster_col].unique())
            
            selected_cluster = st.selectbox("Select Cluster/Topic for Word Cloud:", clusters_list)
            
            if st.button("Generate Word Cloud"):
                cluster_data = df_results[df_results[cluster_col] == selected_cluster]
                cluster_text = ' '.join(cluster_data['processed_review'].astype(str))
                
                if cluster_text.strip():
                    # Generate word cloud
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white',
                        max_words=100,
                        colormap='viridis'
                    ).generate(cluster_text)
                    
                    # Display word cloud
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title(f'Word Cloud for {cluster_col.title()} {selected_cluster}')
                    
                    st.pyplot(fig)
                else:
                    st.warning("No text data available for this cluster/topic.")
        else:
            st.info("Run clustering analysis first to generate word clouds.")
    
    with tab4:
        st.markdown("### 📈 Advanced Visualizations")
        
        if 'clustering_results' in st.session_state:
            df_results = st.session_state['clustering_results']
            method = st.session_state['clustering_method']
            
            # Rating analysis by cluster/topic
            if 'Rating' in df_results.columns:
                cluster_col = 'cluster' if method == 'kmeans' else 'topic'
                
                st.markdown("#### ⭐ Rating Distribution by Cluster/Topic")
                
                fig_box = px.box(
                    df_results, 
                    x=cluster_col, 
                    y='Rating',
                    title=f"Rating Distribution by {cluster_col.title()}"
                )
                st.plotly_chart(fig_box, use_container_width=True)
                
                # Average ratings
                avg_ratings = df_results.groupby(cluster_col)['Rating'].mean().reset_index()
                
                fig_bar = px.bar(
                    avg_ratings, 
                    x=cluster_col, 
                    y='Rating',
                    title=f"Average Rating by {cluster_col.title()}"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Company distribution
            if 'Company Name' in df_results.columns:
                st.markdown("#### 🏢 Company Distribution")
                
                cluster_col = 'cluster' if method == 'kmeans' else 'topic'
                
                # Top companies by cluster
                company_cluster = df_results.groupby(['Company Name', cluster_col]).size().reset_index(name='count')
                top_companies = company_cluster.groupby('Company Name')['count'].sum().nlargest(10).index
                
                company_cluster_filtered = company_cluster[company_cluster['Company Name'].isin(top_companies)]
                
                fig_stacked = px.bar(
                    company_cluster_filtered,
                    x='Company Name',
                    y='count',
                    color=cluster_col,
                    title=f"Top Companies by {cluster_col.title()}",
                    text='count'
                )
                fig_stacked.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_stacked, use_container_width=True)
        else:
            st.info("Run clustering analysis first to see visualizations.")

else:
    st.error("❌ Data not loaded. Please check the data file.")

# Information section
st.markdown("---")
st.markdown("## 📚 About Information Clustering")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🎯 K-Means Clustering
    
    K-Means groups reviews based on content similarity:
    
    - **Distance-based**: Groups similar content together
    - **Fixed clusters**: Pre-defined number of clusters
    - **Hard assignment**: Each review belongs to one cluster
    - **Fast**: Efficient for large datasets
    """)

with col2:
    st.markdown("""
    ### 🔍 LDA Topic Modeling
    
    LDA discovers hidden topics in review collections:
    
    - **Probabilistic**: Reviews can belong to multiple topics
    - **Interpretable**: Topics defined by word distributions
    - **Flexible**: Automatic topic discovery
    - **Contextual**: Captures semantic relationships
    """)

st.markdown("""
### 💡 Business Applications

**For Companies:**
- 🎯 Identify key improvement areas
- 📊 Understand employee sentiment patterns  
- 🔍 Benchmark against industry standards
- 💡 Generate actionable insights

**For ITViec:**
- 📈 Provide value-added analytics services
- 🤝 Help companies improve workplace culture
- 📊 Offer data-driven consulting
- 🚀 Differentiate platform capabilities
""")
