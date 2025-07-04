
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering # Import other clustering methods
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import silhouette_score, davies_bouldin_score # Import Davies-Bouldin Score
from tqdm.auto import tqdm # Import tqdm for progress bar

# Assuming the helper functions (evaluate_clustering, plot_pca_clusters, calculate_wcss, plot_wcss, get_top_n_words, analyze_clusters) are defined in previous cells.

def run_clustering_pipeline_for_column(file_path, text_column, n_topics=3, n_clusters=3, k_range_optimal=range(2, 6), dbscan_eps=0.5, dbscan_min_samples=5, pca_components_dbscan=2):
    """
    Runs the clustering analysis pipeline for a specified text column.

    Args:
        file_path (str): The path to the Excel file.
        text_column (str): The name of the column containing the text data for analysis.
        n_topics (int): The desired number of topics for LDA.
        n_clusters (int): The desired number of clusters for KMeans and Agglomerative.
        k_range_optimal (range): The range of cluster numbers to evaluate for optimal k.
        dbscan_eps (float): The maximum distance between two samples for DBSCAN.
        dbscan_min_samples (int): The number of samples in a neighborhood for DBSCAN.
        pca_components_dbscan (int): Number of PCA components for DBSCAN.

    Returns:
        pd.DataFrame: The DataFrame with added clustering and topic information for the specified column.
        LatentDirichletAllocation: The fitted LDA model.
        CountVectorizer: The fitted vectorizer.
        dict: A dictionary containing evaluation metrics.
    """
    pipeline = ClusteringAnalysisPipeline(file_path)

    # Load and preprocess data
    pipeline.load_and_preprocess_data(text_column=text_column)

    if pipeline.df is None:
        return None, None, None, None

    # Perform count vectorization
    pipeline.perform_count_vectorization(text_column=text_column)

    if pipeline.doc_term_matrix is None:
        return pipeline.df, None, None, None

    # Apply LDA topic modeling
    pipeline.apply_lda(n_topics=n_topics)

    if pipeline.lda_matrix is None:
        return pipeline.df, None, None, None

    # Determine the optimal number of clusters (optional, can be run separately)
    print(f"\nDetermining optimal clusters for '{text_column}':")
    pipeline.determine_optimal_clusters(k_range=k_range_optimal)

    # Apply clustering
    pipeline.apply_clustering(
        n_clusters=n_clusters,
        apply_kmeans=True,
        apply_agglomerative=True,
        apply_dbscan=True,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        pca_components_dbscan=pca_components_dbscan
    )

    # Add dominant topic indices and cluster labels to the DataFrame
    if pipeline.dominant_topics is not None:
        pipeline.df['dominant_topic'] = pipeline.dominant_topics
    if pipeline.cluster_labels_kmeans is not None:
        pipeline.df['kmeans_label'] = pipeline.cluster_labels_kmeans
    if pipeline.cluster_labels_agg is not None:
        pipeline.df['agglomerative_label'] = pipeline.cluster_labels_agg
    if pipeline.cluster_labels_dbscan is not None:
        pipeline.df['dbscan_label'] = pipeline.cluster_labels_dbscan

    # Evaluate clustering results
    evaluation_metrics = {}
    if pipeline.cluster_labels_kmeans is not None:
        kmeans_silhouette = evaluate_clustering(pipeline.lda_matrix, pipeline.cluster_labels_kmeans)
        evaluation_metrics['kmeans_silhouette'] = kmeans_silhouette
        print(f"Silhouette Score for KMeans ('{text_column}'): {kmeans_silhouette:.4f}" if kmeans_silhouette is not None else f"Silhouette Score for KMeans ('{text_column}'): N/A")

        try:
            davies_bouldin_kmeans = davies_bouldin_score(pipeline.lda_matrix, pipeline.cluster_labels_kmeans)
            evaluation_metrics['davies_bouldin_kmeans'] = davies_bouldin_kmeans
            print(f"Davies-Bouldin Score for KMeans ('{text_column}'): {davies_bouldin_kmeans:.4f}")
        except Exception as e:
             print(f"Could not calculate Davies-Bouldin score for KMeans ('{text_column}'): {e}")

    if pipeline.cluster_labels_agg is not None:
         data_dense_agg = pipeline.lda_matrix.toarray() if hasattr(pipeline.lda_matrix, 'toarray') else pipeline.lda_matrix
         agg_silhouette = evaluate_clustering(data_dense_agg, pipeline.cluster_labels_agg)
         evaluation_metrics['agg_silhouette'] = agg_silhouette
         print(f"Silhouette Score for Agglomerative ('{text_column}'): {agg_silhouette:.4f}" if agg_silhouette is not None else f"Silhouette Score for Agglomerative ('{text_column}'): N/A")

         try:
            davies_bouldin_agg = davies_bouldin_score(data_dense_agg, pipeline.cluster_labels_agg)
            evaluation_metrics['davies_bouldin_agg'] = davies_bouldin_agg
            print(f"Davies-Bouldin Score for Agglomerative ('{text_column}'): {davies_bouldin_agg:.4f}")
         except Exception as e:
             print(f"Could not calculate Davies-Bouldin score for Agglomerative ('{text_column}'): {e}")

    if pipeline.cluster_labels_dbscan is not None:
         pca_eval_dbscan = PCA(n_components=pca_components_dbscan, random_state=42)
         data_pca_eval_dbscan = pca_eval_dbscan.fit_transform(pipeline.lda_matrix)
         dbscan_silhouette = evaluate_clustering(data_pca_eval_dbscan, pipeline.cluster_labels_dbscan)
         evaluation_metrics['dbscan_silhouette'] = dbscan_silhouette
         print(f"Silhouette Score for DBSCAN ('{text_column}'): {dbscan_silhouette:.4f}" if dbscan_silhouette is not None else f"Silhouette Score for DBSCAN ('{text_column}'): N/A")

         non_noise_indices = pipeline.cluster_labels_dbscan != -1
         if non_noise_indices.sum() > 1 and len(set(pipeline.cluster_labels_dbscan[non_noise_indices])) > 1:
            try:
                davies_bouldin_dbscan = davies_bouldin_score(data_pca_eval_dbscan[non_noise_indices], pipeline.cluster_labels_dbscan[non_noise_indices])
                evaluation_metrics['davies_bouldin_dbscan'] = davies_bouldin_dbscan
                print(f"Davies-Bouldin Score for DBSCAN (excluding noise) ('{text_column}'): {davies_bouldin_dbscan:.4f}")
            except Exception as e:
                print(f"Could not calculate Davies-Bouldin score for DBSCAN ('{text_column}'): {e}")
         else:
            print(f"Could not calculate Davies-Bouldin score for DBSCAN ('{text_column}'): Too few non-noise points or clusters.")

    # Visualize clusters
    if 'kmeans_label' in pipeline.df.columns:
        pipeline.visualize_clusters(cluster_label_col='kmeans_label', title_suffix=f"KMeans ('{text_column}')")
    if 'agglomerative_label' in pipeline.df.columns:
        pipeline.visualize_clusters(cluster_label_col='agglomerative_label', title_suffix=f"Agglomerative ('{text_column}')")
    if 'dbscan_label' in pipeline.df.columns:
         pipeline.visualize_clusters(cluster_label_col='dbscan_label', title_suffix=f"DBSCAN ('{text_column}')")

    # Analyze cluster characteristics
    if 'kmeans_label' in pipeline.df.columns:
        pipeline.analyze_cluster_characteristics(cluster_label_col='kmeans_label', text_col=text_column)
    if 'agglomerative_label' in pipeline.df.columns:
        pipeline.analyze_cluster_characteristics(cluster_label_col='agglomerative_label', text_col=text_column)
    if 'dbscan_label' in pipeline.df.columns:
        pipeline.analyze_cluster_characteristics(cluster_label_col='dbscan_label', text_col=text_column)

    return pipeline.df, pipeline.lda_model, pipeline.vectorizer, evaluation_metrics
