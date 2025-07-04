"""
Visualization utilities for sentiment analysis and clustering
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def plot_sentiment_distribution(df, sentiment_col='sentiment', title="Sentiment Distribution"):
    """
    Plot sentiment distribution
    
    Args:
        df (pd.DataFrame): Input dataframe
        sentiment_col (str): Name of sentiment column
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    if sentiment_col not in df.columns:
        print(f"Column '{sentiment_col}' not found")
        return None
    
    sentiment_counts = df[sentiment_col].value_counts()
    
    # Define colors for sentiments
    colors = {
        'positive': '#28a745',
        'negative': '#dc3545',
        'neutral': '#ffc107'
    }
    
    # Create pie chart
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title=title,
        color=sentiment_counts.index,
        color_discrete_map=colors
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels (list): Label names
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    if labels is None:
        labels = ['Negative', 'Neutral', 'Positive']
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Create heatmap
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels,
        y=labels,
        title=title,
        text_auto=True,
        color_continuous_scale='Blues'
    )
    
    return fig


def plot_model_comparison(results_df, metric='Accuracy', title="Model Performance Comparison"):
    """
    Plot model comparison results
    
    Args:
        results_df (pd.DataFrame): Model comparison results
        metric (str): Metric to plot
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    if metric not in results_df.columns:
        print(f"Metric '{metric}' not found")
        return None
    
    # Sort by metric
    results_sorted = results_df.sort_values(metric, ascending=True)
    
    fig = px.bar(
        results_sorted,
        x=metric,
        y='Model',
        orientation='h',
        title=title,
        color=metric,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=400)
    
    return fig


def plot_feature_importance(feature_df, top_n=20, title="Feature Importance"):
    """
    Plot feature importance
    
    Args:
        feature_df (pd.DataFrame): Feature importance dataframe
        top_n (int): Number of top features to show
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    if feature_df is None or len(feature_df) == 0:
        print("No feature importance data available")
        return None
    
    # Take top N features
    top_features = feature_df.head(top_n)
    
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title=title,
        color='importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=max(400, top_n * 20))
    
    return fig


def plot_text_length_distribution(df, text_col='processed_text', title="Text Length Distribution"):
    """
    Plot text length distribution
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_col (str): Name of text column
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    if text_col not in df.columns:
        print(f"Column '{text_col}' not found")
        return None
    
    # Calculate text lengths
    text_lengths = df[text_col].str.len()
    
    fig = px.histogram(
        x=text_lengths,
        nbins=50,
        title=title,
        labels={'x': 'Text Length (characters)', 'y': 'Frequency'}
    )
    
    return fig


def plot_clustering_results(df, cluster_col='cluster', x_col='PC1', y_col='PC2', 
                          title="Clustering Results"):
    """
    Plot clustering results
    
    Args:
        df (pd.DataFrame): Input dataframe with cluster assignments
        cluster_col (str): Name of cluster column
        x_col (str): X-axis column (e.g., first principal component)
        y_col (str): Y-axis column (e.g., second principal component)
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    required_cols = [cluster_col, x_col, y_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        return None
    
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=cluster_col.astype(str) if cluster_col in df.columns else None,
        title=title,
        labels={x_col: 'First Principal Component', y_col: 'Second Principal Component'}
    )
    
    return fig


def create_wordcloud(text_data, title="Word Cloud", width=800, height=400):
    """
    Create word cloud visualization
    
    Args:
        text_data (str or list): Text data for word cloud
        title (str): Plot title
        width (int): Word cloud width
        height (int): Word cloud height
        
    Returns:
        matplotlib.figure.Figure: Matplotlib figure
    """
    if isinstance(text_data, list):
        text = ' '.join(text_data)
    else:
        text = str(text_data)
    
    if not text.strip():
        print("No text data provided")
        return None
    
    # Create word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate(text)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, pad=20)
    
    plt.tight_layout()
    return fig


def plot_sentiment_over_time(df, date_col, sentiment_col='sentiment', 
                           title="Sentiment Trends Over Time"):
    """
    Plot sentiment trends over time
    
    Args:
        df (pd.DataFrame): Input dataframe
        date_col (str): Name of date column
        sentiment_col (str): Name of sentiment column
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    required_cols = [date_col, sentiment_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        return None
    
    # Convert date column if needed
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    df_copy = df_copy.dropna(subset=[date_col])
    
    # Group by date and sentiment
    sentiment_by_date = df_copy.groupby([df_copy[date_col].dt.date, sentiment_col]).size().reset_index(name='count')
    
    fig = px.line(
        sentiment_by_date,
        x=date_col,
        y='count',
        color=sentiment_col,
        title=title,
        color_discrete_map={
            'positive': '#28a745',
            'negative': '#dc3545',
            'neutral': '#ffc107'
        }
    )
    
    return fig


def plot_rating_distribution(df, rating_col='Rating', title="Rating Distribution"):
    """
    Plot rating distribution
    
    Args:
        df (pd.DataFrame): Input dataframe
        rating_col (str): Name of rating column
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    if rating_col not in df.columns:
        print(f"Column '{rating_col}' not found")
        return None
    
    rating_counts = df[rating_col].value_counts().sort_index()
    
    fig = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        title=title,
        labels={'x': 'Rating', 'y': 'Count'},
        color=rating_counts.values,
        color_continuous_scale='Viridis'
    )
    
    return fig


def plot_correlation_matrix(df, columns=None, title="Correlation Matrix"):
    """
    Plot correlation matrix
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): Columns to include in correlation
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    if columns is None:
        # Use only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            print("Not enough numeric columns for correlation")
            return None
        columns = numeric_cols
    
    # Calculate correlation matrix
    corr_matrix = df[columns].corr()
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        title=title,
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    
    return fig


def plot_cluster_characteristics(df, cluster_col='cluster', feature_cols=None, 
                                title="Cluster Characteristics"):
    """
    Plot cluster characteristics using radar chart
    
    Args:
        df (pd.DataFrame): Input dataframe
        cluster_col (str): Name of cluster column
        feature_cols (list): Features to include in radar chart
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    if cluster_col not in df.columns:
        print(f"Column '{cluster_col}' not found")
        return None
    
    if feature_cols is None:
        # Use numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != cluster_col]
    
    if len(feature_cols) < 3:
        print("Need at least 3 features for radar chart")
        return None
    
    # Calculate cluster means
    cluster_means = df.groupby(cluster_col)[feature_cols].mean()
    
    # Normalize to 0-1 scale
    cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
    
    fig = go.Figure()
    
    for cluster in cluster_means_norm.index:
        fig.add_trace(go.Scatterpolar(
            r=cluster_means_norm.loc[cluster].values,
            theta=feature_cols,
            fill='toself',
            name=f'Cluster {cluster}'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title=title
    )
    
    return fig


def create_dashboard_summary(df, sentiment_col='sentiment', rating_col='Rating'):
    """
    Create summary statistics for dashboard
    
    Args:
        df (pd.DataFrame): Input dataframe
        sentiment_col (str): Name of sentiment column
        rating_col (str): Name of rating column
        
    Returns:
        dict: Summary statistics
    """
    summary = {
        'total_reviews': len(df),
        'sentiment_distribution': {},
        'average_rating': 0,
        'rating_distribution': {},
        'top_companies': []
    }
    
    # Sentiment distribution
    if sentiment_col in df.columns:
        sentiment_counts = df[sentiment_col].value_counts()
        summary['sentiment_distribution'] = sentiment_counts.to_dict()
    
    # Rating statistics
    if rating_col in df.columns:
        summary['average_rating'] = df[rating_col].mean()
        rating_counts = df[rating_col].value_counts().sort_index()
        summary['rating_distribution'] = rating_counts.to_dict()
    
    # Top companies
    if 'Company Name' in df.columns:
        top_companies = df['Company Name'].value_counts().head(10)
        summary['top_companies'] = top_companies.to_dict()
    
    return summary
