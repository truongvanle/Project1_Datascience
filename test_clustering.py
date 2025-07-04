#!/usr/bin/env python3
"""
Test script to verify clustering functionality works end-to-end
"""

import sys
import os
sys.path.append('.')
os.chdir('/Users/ed/it_viec_project1')

print("🔍 Testing Information Clustering Functionality")
print("=" * 50)

try:
    # Import required modules
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA, LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    import warnings
    warnings.filterwarnings('ignore')
    
    print("✅ All required modules imported successfully")
    
    # Test data loading (simulating the clustering page function)
    def load_clustering_data():
        local_data_path = "data/final_data.xlsx"
        external_data_path = "Project1/final_data.xlsx"
        
        for data_path in [local_data_path, external_data_path]:
            if os.path.exists(data_path):
                try:
                    df = pd.read_excel(data_path)
                    # Check for processed text column
                    if 'processed_review' in df.columns:
                        df = df.dropna(subset=['processed_review'])
                        return df
                    elif 'processed' in df.columns:
                        df = df.dropna(subset=['processed'])
                        return df
                    else:
                        df['processed_review'] = df.apply(lambda row: 
                            ' '.join([str(row.get('Title', '')), 
                                    str(row.get('What I liked', '')), 
                                    str(row.get('Suggestions for improvement', ''))]), axis=1)
                        df = df.dropna(subset=['processed_review'])
                        return df
                except Exception as e:
                    continue
        return None
    
    # Load data
    print("\n📊 Loading clustering data...")
    df = load_clustering_data()
    
    if df is None:
        print("❌ Failed to load data")
        exit(1)
    
    print(f"✅ Data loaded: {df.shape[0]} reviews with {df.shape[1]} columns")
    
    # Test K-means clustering
    print("\n🔬 Testing K-means clustering...")
    texts = df['processed_review'].tolist()[:500]  # Use 500 samples
    
    # Vectorization
    vectorizer = TfidfVectorizer(
        max_features=1000, 
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X = vectorizer.fit_transform(texts)
    print(f"   Vectorized to {X.shape} matrix")
    
    # K-means clustering
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    unique, counts = np.unique(clusters, return_counts=True)
    print(f"   Created {len(unique)} clusters: {dict(zip(unique, counts))}")
    
    # Test PCA visualization
    print("\n📈 Testing PCA for visualization...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X.toarray())
    print(f"   PCA reduced to {X_pca.shape} for visualization")
    print(f"   Explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Test LDA topic modeling
    print("\n📚 Testing LDA topic modeling...")
    count_vectorizer = CountVectorizer(
        max_features=100, 
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X_count = count_vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(
        n_components=5, 
        random_state=42, 
        max_iter=10,
        learning_method='batch'
    )
    lda_matrix = lda.fit_transform(X_count)
    print(f"   LDA topics matrix: {lda_matrix.shape}")
    
    # Show top words for each topic
    feature_names = count_vectorizer.get_feature_names_out()
    print("   Top words per topic:")
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-5:][::-1]]
        print(f"     Topic {topic_idx}: {', '.join(top_words)}")
    
    # Test cluster analysis
    print("\n🔍 Testing cluster analysis...")
    df_sample = df.iloc[:500].copy()
    df_sample['cluster'] = clusters
    
    for cluster_id in range(n_clusters):
        cluster_data = df_sample[df_sample['cluster'] == cluster_id]
        if len(cluster_data) > 0:
            avg_rating = cluster_data['Rating'].mean()
            print(f"   Cluster {cluster_id}: {len(cluster_data)} reviews, avg rating: {avg_rating:.2f}")
    
    print("\n🎉 All clustering functionality tests passed!")
    print("✅ Information Clustering is ready to use!")
    
except Exception as e:
    print(f"\n❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
