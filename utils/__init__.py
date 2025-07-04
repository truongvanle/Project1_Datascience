"""
ITViec Analytics Platform - Utility Modules

This package contains utility functions for:
- Text preprocessing and NLP operations
- Machine learning model utilities
- Visualization and plotting functions
"""

__version__ = "1.0.0"
__author__ = "Đào Tuấn Thịnh, Trương Văn Lê"

from .preprocessing import (
    preprocess_text,
    advanced_preprocess_text,
    load_and_preprocess_data,
    create_sentiment_labels,
    balance_dataset,
    extract_features
)

from .model_utils import (
    prepare_features,
    train_sentiment_model,
    compare_models,
    save_model,
    load_model,
    predict_sentiment,
    get_feature_importance,
    evaluate_model_performance,
    hyperparameter_tuning
)

from .visualization import (
    plot_sentiment_distribution,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_feature_importance,
    plot_text_length_distribution,
    plot_clustering_results,
    create_wordcloud,
    plot_sentiment_over_time,
    plot_rating_distribution,
    plot_correlation_matrix,
    plot_cluster_characteristics,
    create_dashboard_summary
)

__all__ = [
    # Preprocessing utilities
    'preprocess_text',
    'advanced_preprocess_text', 
    'load_and_preprocess_data',
    'create_sentiment_labels',
    'balance_dataset',
    'extract_features',
    
    # Model utilities
    'prepare_features',
    'train_sentiment_model',
    'compare_models',
    'save_model',
    'load_model',
    'predict_sentiment',
    'get_feature_importance',
    'evaluate_model_performance',
    'hyperparameter_tuning',
    
    # Visualization utilities
    'plot_sentiment_distribution',
    'plot_confusion_matrix',
    'plot_model_comparison',
    'plot_feature_importance',
    'plot_text_length_distribution',
    'plot_clustering_results',
    'create_wordcloud',
    'plot_sentiment_over_time',
    'plot_rating_distribution',
    'plot_correlation_matrix',
    'plot_cluster_characteristics',
    'create_dashboard_summary'
]
