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

class ClusteringAnalysisPipeline:
    """
    A pipeline for performing clustering analysis on text data, including
    vectorization, LDA topic modeling, clustering, evaluation, and visualization.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.vectorizer = None
        self.doc_term_matrix = None
        self.lda_model = None
        self.lda_matrix = None
        self.dominant_topics = None
        self.kmeans_model = None
        self.cluster_labels_kmeans = None
        self.cluster_labels_agg = None # To store Agglomerative clustering labels
        self.cluster_labels_dbscan = None # To store DBSCAN labels
        self.pca_coords = None

    def load_and_preprocess_data(self, text_column='processed'):
        """
        Loads data from an Excel file and preprocesses it by dropping rows with missing
        values in the specified text column.

        Args:
            text_column (str): The name of the column containing the text data.
        """
        try:
            self.df = pd.read_excel(self.file_path)
            self.df.dropna(subset=[text_column], inplace=True)
            print("Data loaded and preprocessed.")
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            self.df = None

    def perform_count_vectorization(self, text_column='processed', max_df=0.95, min_df=2):
        """
        Performs Count Vectorization on a specified text column of the DataFrame.

        Args:
            text_column (str): The name of the column containing the text data.
            max_df (float): Ignore terms that appear in documents more than the specified threshold.
            min_df (float or int): Ignore terms that appear in documents less than the specified threshold.
        """
        if self.df is not None:
            self.vectorizer = CountVectorizer(max_df=max_df, min_df=min_df)
            self.doc_term_matrix = self.vectorizer.fit_transform(self.df[text_column])
            print(f"Count Vectorization performed. Document-term matrix shape: {self.doc_term_matrix.shape}")
        else:
            print("DataFrame is not loaded. Cannot perform vectorization.")

    def apply_lda(self, n_topics):
        """
        Applies Latent Dirichlet Allocation (LDA) to a document-term matrix
        and determines the dominant topic for each document.

        Args:
            n_topics (int): The desired number of topics.
        """
        if self.doc_term_matrix is not None:
            self.lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            self.lda_matrix = self.lda_model.fit_transform(self.doc_term_matrix)
            self.dominant_topics = np.argmax(self.lda_matrix, axis=1)
            self.df['dominant_topic'] = self.dominant_topics
            print(f"LDA applied with {n_topics} topics.")
        else:
            print("Document-term matrix is not available. Cannot apply LDA.")

    def apply_clustering(self, n_clusters=3, apply_kmeans=True, apply_agglomerative=True, apply_dbscan=False, dbscan_eps=0.5, dbscan_min_samples=5, pca_components_dbscan=50):
        """
        Applies KMeans, Agglomerative Clustering, and/or DBSCAN to the data.

        Args:
            n_clusters (int): The desired number of clusters for KMeans and Agglomerative.
            apply_kmeans (bool): Whether to apply KMeans clustering.
            apply_agglomerative (bool): Whether to apply Agglomerative Clustering.
            apply_dbscan (bool): Whether to apply DBSCAN clustering.
            dbscan_eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other for DBSCAN.
            dbscan_min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point for DBSCAN.
            pca_components_dbscan (int): Number of PCA components to use before applying DBSCAN.
        """
        if self.lda_matrix is not None or self.doc_term_matrix is not None:
            data_for_clustering = self.lda_matrix if self.lda_matrix is not None else self.doc_term_matrix

            if apply_kmeans:
                self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                self.cluster_labels_kmeans = self.kmeans_model.fit_predict(data_for_clustering)
                self.df['kmeans_label'] = self.cluster_labels_kmeans
                print(f"KMeans clustering applied with {n_clusters} clusters.")

            if apply_agglomerative:
                # Agglomerative Clustering typically works better on dense data
                data_dense = data_for_clustering.toarray() if hasattr(data_for_clustering, 'toarray') else data_for_clustering
                agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
                self.cluster_labels_agg = agg_clustering.fit_predict(data_dense)
                self.df['agglomerative_label'] = self.cluster_labels_agg
                print(f"Agglomerative Clustering applied with {n_clusters} clusters.")

            if apply_dbscan:
                 # Apply PCA before DBSCAN for dimensionality reduction
                data_dense = data_for_clustering.toarray() if hasattr(data_for_clustering, 'toarray') else data_for_clustering
                pca_dbscan = PCA(n_components=pca_components_dbscan, random_state=42)
                data_pca_dbscan = pca_dbscan.fit_transform(data_dense)

                print(f"Attempting DBSCAN on PCA-reduced data ({pca_components_dbscan} components). This might take time.")
                dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
                self.cluster_labels_dbscan = dbscan.fit_predict(data_pca_dbscan)
                self.df['dbscan_label'] = self.cluster_labels_dbscan
                print("DBSCAN clustering applied.")

        else:
            print("Data for clustering is not available. Cannot apply clustering.")


    def get_topic_keywords(self, n_top_words=10):
        """
        Gets the top keywords for each topic from the fitted LDA model.

        Args:
            n_top_words (int): The number of top keywords to retrieve for each topic.

        Returns:
            list: A list of strings, where each string contains the top keywords for a topic.
        """
        if self.lda_model is not None and self.vectorizer is not None:
            feature_names = self.vectorizer.get_feature_names_out()
            topic_keywords = []
            for topic_idx, topic in enumerate(self.lda_model.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
                topic_keywords.append(f"Chủ đề #{topic_idx + 1}: {', '.join(top_words)}")
            if self.df is not None:
                 self.df['dominant_topic_keywords'] = self.df['dominant_topic'].apply(lambda x: topic_keywords[x])
            return topic_keywords
        else:
            print("LDA model or vectorizer not available. Cannot get topic keywords.")
            return []

    def get_cluster_keywords(self, cluster_label_col, n_top_words=10):
        """
        Gets the top keywords for each cluster based on word frequency within each cluster.

        Args:
            cluster_label_col (str): The name of the column containing cluster labels.
            n_top_words (int): The number of top keywords to retrieve for each cluster.

        Returns:
            dict: A dictionary where keys are cluster labels and values are lists of top keywords.
        """
        if self.df is not None and self.vectorizer is not None and cluster_label_col in self.df.columns:
            feature_names = self.vectorizer.get_feature_names_out()
            cluster_keywords = {}
            for cluster_id in sorted(self.df[cluster_label_col].unique()):
                cluster_df = self.df[self.df[cluster_label_col] == cluster_id]
                if not cluster_df.empty:
                    # Get the document-term matrix for the reviews in this cluster
                    cluster_doc_term_matrix = self.vectorizer.transform(cluster_df['processed'])
                    # Sum the word counts across all documents in the cluster
                    sum_words = cluster_doc_term_matrix.sum(axis=0)
                    # Get the indices of the top words
                    top_word_indices = sum_words.argsort()[0, ::-1][:n_top_words]
                    # Get the actual top words
                    top_words = [feature_names[i] for i in top_word_indices.tolist()[0]]
                    cluster_keywords[cluster_id] = top_words
                else:
                    cluster_keywords[cluster_id] = []

            if self.df is not None:
                 self.df[f'{cluster_label_col}_keywords'] = self.df[cluster_label_col].apply(lambda x: ', '.join(cluster_keywords[x]))
            return cluster_keywords

        else:
            print("DataFrame, vectorizer, or specified cluster label column not available. Cannot get cluster keywords.")
            return {}

    def evaluate_clustering(self, data_for_evaluation, labels, clustering_method_name):
        """
        Calculates and prints evaluation metrics (Silhouette, Davies-Bouldin) for clustering results.

        Args:
            data_for_evaluation (numpy.ndarray or scipy.sparse.csr_matrix): The data matrix used for clustering.
            labels (numpy.ndarray): The cluster labels.
            clustering_method_name (str): The name of the clustering method (e.g., 'KMeans', 'Agglomerative', 'DBSCAN').
        """
        print(f"\nEvaluating {clustering_method_name} Clustering:")
        # Calculate Silhouette Score
        silhouette = evaluate_clustering(data_for_evaluation, labels) # Using the existing function
        if silhouette is not None:
            print(f"  Silhouette Score: {silhouette:.4f}")

        # Calculate Davies-Bouldin Score
        # Davies-Bouldin requires dense data and at least 2 clusters and no noise points (-1)
        if hasattr(data_for_evaluation, 'toarray'):
             data_dense = data_for_evaluation.toarray()
        else:
             data_dense = data_for_evaluation

        # Filter out noise points (-1) and check for sufficient clusters
        if clustering_method_name == 'DBSCAN':
            non_noise_indices = labels != -1
            if non_noise_indices.sum() > 1 and len(set(labels[non_noise_indices])) > 1:
                try:
                    davies_bouldin = davies_bouldin_score(data_dense[non_noise_indices], labels[non_noise_indices])
                    print(f"  Davies-Bouldin Score (excluding noise): {davies_bouldin:.4f}")
                except Exception as e:
                    print(f"  Could not calculate Davies-Bouldin score: {e}")
            else:
                print("  Could not calculate Davies-Bouldin score for DBSCAN: Too few non-noise points or clusters.")

        else: # For KMeans and Agglomerative
            if len(set(labels)) > 1:
                try:
                    davies_bouldin = davies_bouldin_score(data_dense, labels)
                    print(f"  Davies-Bouldin Score: {davies_bouldin:.4f}")
                except Exception as e:
                    print(f"  Could not calculate Davies-Bouldin score: {e}")
            else:
                 print("  Could not calculate Davies-Bouldin score: Too few clusters.")


    def determine_optimal_clusters(self, k_range=range(2, 11)):
        """
        Determines the optimal number of clusters using the Elbow method (WCSS)
        and Silhouette score.

        Args:
            k_range (range): The range of cluster numbers to evaluate.
        """
        if self.lda_matrix is not None:
            data_for_evaluation = self.lda_matrix
            print("\nDetermining optimal number of clusters for KMeans:")

            # Compute Silhouette scores
            silhouette_scores = []
            print("  Computing Silhouette Scores...")
            for k in tqdm(k_range, desc="    Progress"):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(data_for_evaluation)
                score = silhouette_score(data_for_evaluation, kmeans.labels_)
                silhouette_scores.append(score)

            # Plot Silhouette scores
            plt.figure(figsize=(10, 6))
            plt.plot(k_range, silhouette_scores, marker='o')
            plt.title('Silhouette Score vs. Number of Clusters (KMeans)')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Silhouette Score')
            plt.xticks(k_range)
            plt.grid(True)
            plt.show()

            # Compute WCSS
            wcss = calculate_wcss(data_for_evaluation, k_range) # Using the existing function

            # Plot WCSS
            plot_wcss(wcss, k_range) # Using the existing function

        else:
            print("LDA matrix not available. Cannot determine optimal clusters.")


    def visualize_clusters(self, cluster_label_col, title_suffix="Clustering Visualization (PCA)"):
        """
        Generates a scatter plot of clusters after PCA dimensionality reduction.

        Args:
            cluster_label_col (str): The name of the column containing cluster labels to visualize.
            title_suffix (str): Suffix for the plot title.
        """
        if self.df is not None and cluster_label_col in self.df.columns and self.lda_matrix is not None:
            # Perform PCA if not already done for visualization
            if self.pca_coords is None:
                pca = PCA(n_components=2, random_state=42)
                self.pca_coords = pca.fit_transform(self.lda_matrix)
                self.df['pca_x'] = self.pca_coords[:, 0]
                self.df['pca_y'] = self.pca_coords[:, 1]

            plot_pca_clusters(self.df, 'pca_x', 'pca_y', cluster_label_col, f"{cluster_label_col.replace('_label', '').capitalize()} {title_suffix}") # Using the existing function
        else:
            print("DataFrame, cluster label column, or LDA matrix not available. Cannot visualize clusters.")

    def analyze_cluster_characteristics(self, cluster_label_col, text_col='processed', rating_col='Rating', positive_col='positive_word_count', negative_col='negative_word_count'):
        """
        Analyzes clusters by finding frequent words and calculating average metrics.

        Args:
            cluster_label_col (str): The name of the column containing cluster labels.
            text_col (str): The name of the column containing the text data.
            rating_col (str): The name of the column containing the rating data.
            positive_col (str): The name of the column containing positive word counts.
            negative_col (str): The name of the column containing negative word counts.
        """
        if self.df is not None and cluster_label_col in self.df.columns:
            analyze_clusters(self.df, cluster_label_col, text_col, rating_col, positive_col, negative_col) # Using the existing function
        else:
             print("DataFrame or specified cluster label column not available. Cannot analyze cluster characteristics.")


from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans # Import KMeans for WCSS calculation
from collections import Counter

# Re-define the helper functions that the pipeline class depends on
def evaluate_clustering(X, labels):
    """
    Calculates the Silhouette score for clustering results.

    Args:
        X (numpy.ndarray or scipy.sparse.csr_matrix): The data matrix used for clustering.
        labels (numpy.ndarray): The cluster labels.

    Returns:
        float: The Silhouette score. Returns None if score cannot be calculated.
    """
    if len(set(labels)) > 1 and len(labels) > 1:
        try:
            score = silhouette_score(X, labels)
            return score
        except Exception as e:
            print(f"Could not calculate Silhouette score: {e}")
            return None
    else:
        print("Could not calculate Silhouette score: Too few clusters or samples.")
        return None

def plot_pca_clusters(df, x_col, y_col, hue_col, title):
    """
    Generates a scatter plot of clusters after PCA dimensionality reduction.

    Args:
        df (pd.DataFrame): The DataFrame with PCA components and cluster labels.
        x_col (str): The name of the column containing PCA component 1.
        y_col (str): The name of the column containing PCA component 2.
        hue_col (str): The name of the column containing cluster labels.
        title (str): The title of the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, palette='Set2', legend='full')
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

def calculate_wcss(X, k_range):
    """
    Calculates the Within-Cluster Sum of Squares (WCSS) for a range of k values.

    Args:
        X (numpy.ndarray or scipy.sparse.csr_matrix): The data matrix.
        k_range (range): The range of cluster numbers to evaluate.

    Returns:
        list: A list of WCSS values for each k in the range.
    """
    wcss = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    return wcss

def plot_wcss(wcss, k_range):
    """
    Plots the WCSS values against the number of clusters (k).

    Args:
        wcss (list): A list of WCSS values.
        k_range (range): The range of cluster numbers evaluated.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()

def get_top_n_words(corpus, n=10):
    """Get the top n most frequent words from a corpus."""
    all_words = ' '.join(corpus).split()
    word_counts = Counter(all_words)
    return word_counts.most_common(n)


def analyze_clusters(df, cluster_label_col, text_col='processed', rating_col='Rating', positive_col='positive_word_count', negative_col='negative_word_count'):
    """
    Analyzes clusters by finding frequent words and calculating average metrics.

    Args:
        df (pd.DataFrame): The DataFrame with cluster labels and text data.
        cluster_label_col (str): The name of the column containing cluster labels.
        text_col (str): The name of the column containing the text data.
        rating_col (str): The name of the column containing the rating data.
        positive_col (str): The name of the column containing positive word counts.
        negative_col (str): The name of the column containing negative word counts.
    """
    print(f"\nAnalyzing clusters for: {cluster_label_col}")
    for cluster_id in sorted(df[cluster_label_col].unique()):
        cluster_df = df[df[cluster_label_col] == cluster_id]

        print(f"\n--- Cluster {cluster_id} ---")

        # Analyze frequent words
        if not cluster_df.empty:
            top_words = get_top_n_words(cluster_df[text_col].dropna().astype(str).tolist())
            print(f"Most frequent words: {', '.join([f'{word} ({count})' for word, count in top_words])}")

            # Analyze sentiment metrics
            # Check if rating_col, positive_col, negative_col exist in the DataFrame
            avg_rating = cluster_df[rating_col].mean() if rating_col in cluster_df.columns else 'N/A'
            avg_positive_words = cluster_df[positive_col].mean() if positive_col in cluster_df.columns else 'N/A'
            avg_negative_words = cluster_df[negative_col].mean() if negative_col in cluster_df.columns else 'N/A'


            print(f"Average Rating: {avg_rating:.2f}" if isinstance(avg_rating, float) else f"Average Rating: {avg_rating}")
            print(f"Average Positive Word Count: {avg_positive_words:.2f}" if isinstance(avg_positive_words, float) else f"Average Positive Word Count: {avg_positive_words}")
            print(f"Average Negative Word Count: {avg_negative_words:.2f}" if isinstance(avg_negative_words, float) else f"Average Negative Word Count: {avg_negative_words}")
            print(f"Number of reviews: {len(cluster_df)}")

        else:
            print("Cluster is empty.")