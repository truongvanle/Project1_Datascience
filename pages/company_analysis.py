import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Page header
st.markdown("""
<div style="background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%); 
            padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
    <h1>🏢 Company Analysis</h1>
    <p style="font-size: 1.2rem; margin-bottom: 0;">Deep Dive into Individual Company Performance & Clustering</p>
</div>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_company_data():
    """Load data for company analysis"""
    # Try different possible data paths
    data_paths = [
        "data/final_data.xlsx",
        "data/reviews.csv", 
        "Project1/final_data.xlsx",
        "data/clustered_reviews.csv"
    ]
    
    for data_path in data_paths:
        if os.path.exists(data_path):
            try:
                if data_path.endswith('.xlsx'):
                    df = pd.read_excel(data_path)
                else:
                    df = pd.read_csv(data_path)
                
                # Ensure required columns exist
                if 'Company Name' in df.columns:
                    # Clean company names
                    df['Company Name'] = df['Company Name'].astype(str).str.strip()
                    df = df[df['Company Name'] != 'nan']
                    df = df[df['Company Name'] != '']
                    return df
                    
            except Exception as e:
                continue
    
    st.error("No suitable data file found for company analysis. Please ensure data files are available.")
    return None

def preprocess_text(text):
    """Preprocess text for analysis"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def combine_review_text(row):
    """Combine all review text columns into one"""
    text_parts = []
    
    # List of potential text columns
    text_columns = ['Title', 'What I liked', 'Suggestions for improvement']
    
    for col in text_columns:
        if col in row.index and pd.notna(row[col]):
            text_parts.append(str(row[col]))
    
    return ' '.join(text_parts)

@st.cache_resource
def perform_clustering_for_company(company_texts, n_clusters=3):
    """Perform clustering specifically for a company's reviews"""
    try:
        if len(company_texts) < n_clusters:
            n_clusters = max(1, len(company_texts))
        
        # Vectorize texts
        vectorizer = TfidfVectorizer(
            max_features=500, 
            stop_words=None, 
            max_df=0.95, 
            min_df=1,
            ngram_range=(1, 2)
        )
        X = vectorizer.fit_transform(company_texts)
        
        if X.shape[0] < n_clusters:
            n_clusters = X.shape[0]
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # PCA for visualization if we have enough samples
        if X.shape[0] > 1:
            pca = PCA(n_components=min(2, X.shape[1]), random_state=42)
            X_pca = pca.fit_transform(X.toarray())
        else:
            X_pca = np.array([[0, 0]])
        
        return clusters, X_pca, vectorizer, kmeans
    except Exception as e:
        st.error(f"Clustering error: {e}")
        return None, None, None, None

def get_cluster_keywords(texts, clusters, vectorizer, n_words=15):
    """Get top keywords for each cluster"""
    cluster_keywords = {}
    
    for cluster_id in np.unique(clusters):
        cluster_texts = [texts[i] for i in range(len(texts)) if clusters[i] == cluster_id]
        if not cluster_texts:
            continue
            
        cluster_text = ' '.join(cluster_texts)
        
        # Get TF-IDF scores
        try:
            tfidf_matrix = vectorizer.transform([cluster_text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top words
            scores = tfidf_matrix.toarray()[0]
            top_indices = scores.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_indices if scores[i] > 0]
            
            cluster_keywords[cluster_id] = top_words
        except:
            cluster_keywords[cluster_id] = []
    
    return cluster_keywords

def create_wordcloud(text, title="Word Cloud"):
    """Create a word cloud for the given text"""
    if not text or not text.strip():
        return None
        
    try:
        # Clean the text a bit more for wordcloud
        cleaned_text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        if len(cleaned_text.split()) < 3:  # Need at least 3 words
            return None
            
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis',
            min_font_size=10,
            max_font_size=100,
            relative_scaling=0.5,
            stopwords=set(['công ty', 'company', 'làm', 'work', 'job', 'việc'])
        ).generate(cleaned_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        return fig
    except Exception as e:
        st.warning(f"Could not generate word cloud: {str(e)}")
        return None

def calculate_overall_score(row):
    """Calculate overall score from rating columns"""
    rating_cols = ['Rating', 'Salary & benefits', 'Training & learning', 
                   'Management cares about me', 'Culture & fun', 'Office & workspace']
    
    scores = []
    for col in rating_cols:
        if col in row.index and pd.notna(row[col]):
            try:
                score = float(row[col])
                if 1 <= score <= 5:  # Valid rating range
                    scores.append(score)
            except:
                continue
    
    return np.mean(scores) if scores else None

# Main application
def main():
    # Load data
    df = load_company_data()
    if df is None:
        return
    
    # Clean and prepare data
    if 'Company Name' in df.columns:
        companies = sorted(df['Company Name'].dropna().unique())
    else:
        st.error("No 'Company Name' column found in the data")
        return
    
    # Company selection
    st.markdown("## 🏢 Select Company for Analysis")
    selected_company = st.selectbox(
        "Choose a company to analyze:",
        options=companies,
        index=0 if companies else None
    )
    
    if not selected_company:
        st.warning("Please select a company to proceed with the analysis.")
        return
    
    # Filter data for selected company
    company_df = df[df['Company Name'] == selected_company].copy()
    
    if company_df.empty:
        st.warning(f"No data found for {selected_company}")
        return
    
    # Display company overview
    st.markdown(f"## 📊 Analysis for **{selected_company}**")
    
    # Company stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", len(company_df))
    
    with col2:
        if 'Rating' in company_df.columns:
            avg_rating = company_df['Rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.2f}/5" if pd.notna(avg_rating) else "N/A")
        else:
            st.metric("Average Rating", "N/A")
    
    with col3:
        if 'Recommend?' in company_df.columns:
            recommend_pct = (company_df['Recommend?'] == 'Yes').mean() * 100
            st.metric("Recommendation Rate", f"{recommend_pct:.1f}%" if pd.notna(recommend_pct) else "N/A")
        else:
            st.metric("Recommendation Rate", "N/A")
    
    with col4:
        # Calculate overall score
        company_df['overall_score'] = company_df.apply(calculate_overall_score, axis=1)
        avg_overall = company_df['overall_score'].mean()
        st.metric("Overall Score", f"{avg_overall:.2f}/5" if pd.notna(avg_overall) else "N/A")
    
    # Prepare text for clustering
    company_df['combined_text'] = company_df.apply(combine_review_text, axis=1)
    company_df['processed_text'] = company_df['combined_text'].apply(preprocess_text)
    
    # Remove empty texts
    valid_texts = company_df[company_df['processed_text'].str.len() > 10]['processed_text'].tolist()
    
    if len(valid_texts) < 2:
        st.warning("Not enough text data for meaningful clustering analysis.")
        return
    
    # Clustering Analysis
    st.markdown("## 🔍 Clustering Analysis")
    
    # Let user choose number of clusters
    max_clusters = min(10, len(valid_texts))
    n_clusters = st.slider("Number of clusters:", min_value=2, max_value=max_clusters, value=min(5, max_clusters))
    
    # Perform clustering
    with st.spinner("Performing clustering analysis..."):
        clusters, X_pca, vectorizer, kmeans = perform_clustering_for_company(valid_texts, n_clusters)
    
    if clusters is not None:
        # Add cluster labels to dataframe
        valid_indices = company_df[company_df['processed_text'].str.len() > 10].index
        company_df.loc[valid_indices, 'cluster'] = clusters
        
        # Cluster distribution
        col1, col2 = st.columns(2)
        
        with col1:
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            fig_pie = px.pie(
                values=cluster_counts.values,
                names=[f"Cluster {i}" for i in cluster_counts.index],
                title="Review Distribution by Cluster"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Cluster visualization
            if X_pca is not None and len(X_pca) > 1:
                fig_scatter = px.scatter(
                    x=X_pca[:, 0],
                    y=X_pca[:, 1],
                    color=[f"Cluster {c}" for c in clusters],
                    title="Cluster Visualization (PCA)",
                    labels={'x': 'PC1', 'y': 'PC2'}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Cluster keywords and word clouds
        st.markdown("## 🏷️ Cluster Keywords & Word Clouds")
        
        cluster_keywords = get_cluster_keywords(valid_texts, clusters, vectorizer)
        
        for cluster_id in sorted(cluster_keywords.keys()):
            with st.expander(f"🔖 Cluster {cluster_id} Analysis", expanded=True):
                # Cluster info
                cluster_size = np.sum(clusters == cluster_id)
                st.write(f"**Reviews in this cluster:** {cluster_size}")
                
                # Keywords
                keywords = cluster_keywords[cluster_id]
                if keywords:
                    st.write("**Top Keywords:**")
                    st.write(", ".join(keywords[:10]))
                    
                    # Create word cloud
                    cluster_texts = [valid_texts[i] for i in range(len(valid_texts)) if clusters[i] == cluster_id]
                    cluster_text = ' '.join(cluster_texts)
                    
                    wordcloud_fig = create_wordcloud(cluster_text, f"Cluster {cluster_id} Word Cloud")
                    if wordcloud_fig:
                        st.pyplot(wordcloud_fig)
                        plt.close()
                
                # Sample reviews from this cluster
                cluster_reviews = company_df[company_df['cluster'] == cluster_id]
                if not cluster_reviews.empty:
                    st.write("**Sample Reviews:**")
                    sample_size = min(3, len(cluster_reviews))
                    for idx, (_, review) in enumerate(cluster_reviews.head(sample_size).iterrows()):
                        with st.container():
                            st.write(f"**Review {idx+1}:**")
                            if 'Title' in review.index and pd.notna(review['Title']):
                                st.write(f"*Title:* {review['Title']}")
                            if 'What I liked' in review.index and pd.notna(review['What I liked']):
                                st.write(f"*Liked:* {review['What I liked'][:200]}...")
                            if 'Rating' in review.index and pd.notna(review['Rating']):
                                st.write(f"*Rating:* {review['Rating']}/5")
                            st.write("---")
    
    # Company comparison
    st.markdown("## 📈 Company Performance Comparison")
    
    # Calculate company statistics for comparison
    if 'overall_score' in company_df.columns:
        # Compare with other companies
        all_companies_stats = df.groupby('Company Name').agg({
            'Rating': 'mean',
            'overall_score': 'mean'
        }).reset_index()
        
        all_companies_stats = all_companies_stats.dropna()
        
        if not all_companies_stats.empty:
            # Rank companies
            all_companies_stats['rating_rank'] = all_companies_stats['Rating'].rank(ascending=False)
            all_companies_stats['overall_rank'] = all_companies_stats['overall_score'].rank(ascending=False)
            
            # Get selected company's rank
            company_stats = all_companies_stats[all_companies_stats['Company Name'] == selected_company]
            
            if not company_stats.empty:
                company_rank = company_stats.iloc[0]
                total_companies = len(all_companies_stats)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Rating Rank", 
                        f"{int(company_rank['rating_rank'])} / {total_companies}",
                        f"{company_rank['Rating']:.2f} avg rating"
                    )
                
                with col2:
                    st.metric(
                        "Overall Score Rank",
                        f"{int(company_rank['overall_rank'])} / {total_companies}",
                        f"{company_rank['overall_score']:.2f} avg score"
                    )
                
                # Comparison chart
                fig_comparison = go.Figure()
                
                # Add all companies
                fig_comparison.add_trace(go.Scatter(
                    x=all_companies_stats['Rating'],
                    y=all_companies_stats['overall_score'],
                    mode='markers',
                    marker=dict(size=8, color='lightblue', opacity=0.6),
                    name='Other Companies',
                    text=all_companies_stats['Company Name'],
                    hovertemplate='%{text}<br>Rating: %{x:.2f}<br>Overall: %{y:.2f}'
                ))
                
                # Highlight selected company
                fig_comparison.add_trace(go.Scatter(
                    x=[company_rank['Rating']],
                    y=[company_rank['overall_score']],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name=selected_company,
                    text=[selected_company],
                    hovertemplate='%{text}<br>Rating: %{x:.2f}<br>Overall: %{y:.2f}'
                ))
                
                fig_comparison.update_layout(
                    title="Company Performance Comparison",
                    xaxis_title="Average Rating",
                    yaxis_title="Average Overall Score",
                    height=500
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Temporal Analysis (if date column exists)
    st.markdown("## 📅 Temporal Analysis")
    
    if 'Cmt_day' in company_df.columns:
        try:
            # Parse dates
            company_df['date'] = pd.to_datetime(company_df['Cmt_day'], errors='coerce')
            company_df_with_dates = company_df[company_df['date'].notna()]
            
            if not company_df_with_dates.empty:
                # Group by month/year for trend analysis
                company_df_with_dates['year_month'] = company_df_with_dates['date'].dt.to_period('M')
                
                monthly_stats = company_df_with_dates.groupby('year_month').agg({
                    'Rating': ['count', 'mean'],
                    'overall_score': 'mean'
                }).reset_index()
                
                monthly_stats.columns = ['year_month', 'review_count', 'avg_rating', 'avg_overall']
                monthly_stats['year_month_str'] = monthly_stats['year_month'].astype(str)
                
                if len(monthly_stats) > 1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_trend = px.line(
                            monthly_stats,
                            x='year_month_str',
                            y='avg_rating',
                            title='Rating Trend Over Time',
                            markers=True
                        )
                        fig_trend.update_layout(xaxis_title="Month", yaxis_title="Average Rating")
                        st.plotly_chart(fig_trend, use_container_width=True)
                    
                    with col2:
                        fig_count = px.bar(
                            monthly_stats,
                            x='year_month_str',
                            y='review_count',
                            title='Review Volume Over Time'
                        )
                        fig_count.update_layout(xaxis_title="Month", yaxis_title="Number of Reviews")
                        st.plotly_chart(fig_count, use_container_width=True)
        except:
            st.info("Date analysis not available for this dataset.")
    
    # Recent Reviews Section
    st.markdown("## 📝 Recent Reviews Sample")
    
    # Show most recent reviews
    if 'Cmt_day' in company_df.columns:
        try:
            company_df_copy = company_df.copy()
            company_df_copy['date'] = pd.to_datetime(company_df_copy['Cmt_day'], errors='coerce')
            date_filtered = company_df_copy[company_df_copy['date'].notna()]
            if not date_filtered.empty:
                recent_reviews = date_filtered.sort_values(['date'], ascending=[False]).head(5)
            else:
                recent_reviews = company_df.head(5)
        except:
            recent_reviews = company_df.head(5)
    else:
        recent_reviews = company_df.head(5)
    
    for idx, (_, review) in enumerate(recent_reviews.iterrows()):
        with st.expander(f"Review {idx+1} - {review.get('Cmt_day', 'Date N/A')}", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if 'Title' in review.index and pd.notna(review['Title']):
                    st.write(f"**Title:** {review['Title']}")
                if 'What I liked' in review.index and pd.notna(review['What I liked']):
                    st.write(f"**What I liked:** {review['What I liked']}")
                if 'Suggestions for improvement' in review.index and pd.notna(review['Suggestions for improvement']):
                    st.write(f"**Suggestions:** {review['Suggestions for improvement']}")
            
            with col2:
                if 'Rating' in review.index and pd.notna(review['Rating']):
                    st.metric("Overall Rating", f"{review['Rating']}/5")
                if 'Recommend?' in review.index and pd.notna(review['Recommend?']):
                    recommend_color = "green" if review['Recommend?'] == 'Yes' else "red"
                    st.markdown(f"**Recommend:** <span style='color:{recommend_color}'>{review['Recommend?']}</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
