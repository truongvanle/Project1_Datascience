"""
Model utilities for sentiment analysis and clustering
"""
import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import warnings
warnings.filterwarnings('ignore')


def prepare_features(df, text_column='processed_text', method='tfidf', max_features=5000):
    """
    Prepare features for machine learning models
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of text column
        method (str): Feature extraction method ('tfidf', 'count')
        max_features (int): Maximum number of features
        
    Returns:
        tuple: (X, vectorizer) - features and fitted vectorizer
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found")
    
    # Remove empty texts
    texts = df[text_column].dropna().astype(str)
    
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=None,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
    elif method == 'count':
        vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words=None,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
    else:
        raise ValueError("Method must be 'tfidf' or 'count'")
    
    # Fit and transform
    X = vectorizer.fit_transform(texts)
    
    return X, vectorizer


def train_sentiment_model(X, y, model_type='random_forest', test_size=0.2):
    """
    Train sentiment analysis model
    
    Args:
        X: Feature matrix
        y: Target labels
        model_type (str): Type of model to train
        test_size (float): Proportion of test set
        
    Returns:
        dict: Trained model and evaluation metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Select model
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == 'naive_bayes':
        model = MultinomialNB()
    else:
        raise ValueError("Unknown model type")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    results = {
        'model': model,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'test_indices': (X_test, y_test)
    }
    
    return results


def compare_models(X, y, models=None, test_size=0.2):
    """
    Compare multiple models
    
    Args:
        X: Feature matrix
        y: Target labels
        models (list): List of model types to compare
        test_size (float): Proportion of test set
        
    Returns:
        pd.DataFrame: Comparison results
    """
    if models is None:
        models = ['random_forest', 'logistic_regression', 'naive_bayes']
    
    results = []
    
    for model_type in models:
        try:
            result = train_sentiment_model(X, y, model_type, test_size)
            
            results.append({
                'Model': model_type.replace('_', ' ').title(),
                'Accuracy': result['accuracy'],
                'CV_Mean': result['cv_mean'],
                'CV_Std': result['cv_std'],
                'Precision': result['classification_report']['weighted avg']['precision'],
                'Recall': result['classification_report']['weighted avg']['recall'],
                'F1_Score': result['classification_report']['weighted avg']['f1-score']
            })
        except Exception as e:
            print(f"Error training {model_type}: {e}")
    
    return pd.DataFrame(results)


def save_model(model, vectorizer, file_path):
    """
    Save trained model and vectorizer
    
    Args:
        model: Trained model
        vectorizer: Fitted vectorizer
        file_path (str): Path to save model
    """
    try:
        model_data = {
            'model': model,
            'vectorizer': vectorizer,
            'version': '1.0',
            'timestamp': pd.Timestamp.now()
        }
        
        joblib.dump(model_data, file_path)
        print(f"Model saved to {file_path}")
        
    except Exception as e:
        print(f"Error saving model: {e}")


def load_model(file_path):
    """
    Load trained model and vectorizer
    
    Args:
        file_path (str): Path to model file
        
    Returns:
        dict: Loaded model data
    """
    try:
        model_data = joblib.load(file_path)
        print(f"Model loaded from {file_path}")
        return model_data
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def predict_sentiment(text, model, vectorizer, label_map=None):
    """
    Predict sentiment for new text
    
    Args:
        text (str): Input text
        model: Trained model
        vectorizer: Fitted vectorizer
        label_map (dict): Mapping from labels to sentiment names
        
    Returns:
        tuple: (prediction, confidence)
    """
    if label_map is None:
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    try:
        # Vectorize text
        text_vector = vectorizer.transform([text])
        
        # Predict
        prediction = model.predict(text_vector)[0]
        
        # Get confidence if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vector)[0]
            confidence = max(probabilities)
        else:
            confidence = 0.8  # Default confidence
        
        # Map to sentiment name
        sentiment = label_map.get(prediction, 'unknown')
        
        return sentiment, confidence
        
    except Exception as e:
        print(f"Error predicting sentiment: {e}")
        return None, None


def get_feature_importance(model, vectorizer, top_n=20):
    """
    Get feature importance from trained model
    
    Args:
        model: Trained model
        vectorizer: Fitted vectorizer
        top_n (int): Number of top features to return
        
    Returns:
        pd.DataFrame: Feature importance dataframe
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            print("Model doesn't support feature importance")
            return None
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Create dataframe
        feature_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        return feature_df
        
    except Exception as e:
        print(f"Error getting feature importance: {e}")
        return None


def evaluate_model_performance(model, X_test, y_test, label_map=None):
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        label_map (dict): Label mapping
        
    Returns:
        dict: Evaluation metrics
    """
    if label_map is None:
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Per-class metrics
    class_metrics = {}
    for label, name in label_map.items():
        if str(label) in report:
            class_metrics[name] = report[str(label)]
    
    evaluation = {
        'accuracy': accuracy,
        'overall_precision': report['weighted avg']['precision'],
        'overall_recall': report['weighted avg']['recall'],
        'overall_f1': report['weighted avg']['f1-score'],
        'class_metrics': class_metrics,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return evaluation


def hyperparameter_tuning(X, y, model_type='random_forest', param_grid=None, cv=5):
    """
    Perform hyperparameter tuning
    
    Args:
        X: Feature matrix
        y: Target labels
        model_type (str): Type of model
        param_grid (dict): Parameter grid for tuning
        cv (int): Cross-validation folds
        
    Returns:
        dict: Best model and parameters
    """
    from sklearn.model_selection import GridSearchCV
    
    # Default parameter grids
    if param_grid is None:
        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            }
            model = RandomForestClassifier(random_state=42)
        elif model_type == 'logistic_regression':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError("Unsupported model type for tuning")
    
    # Grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    results = {
        'best_model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }
    
    return results
