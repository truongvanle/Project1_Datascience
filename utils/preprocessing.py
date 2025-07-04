"""
Text preprocessing utilities for Vietnamese sentiment analysis
"""
import re
import pandas as pd
import numpy as np
from underthesea import word_tokenize, pos_tag, sent_tokenize


def preprocess_text(text):
    """
    Preprocess Vietnamese text for sentiment analysis
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Cleaned and processed text
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep Vietnamese characters
    text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove very short texts (less than 3 characters)
    if len(text) < 3:
        return ""
    
    return text


def advanced_preprocess_text(text, remove_stopwords=False, use_pos_tags=False):
    """
    Advanced preprocessing with optional stopword removal and POS tagging
    
    Args:
        text (str): Raw text input
        remove_stopwords (bool): Whether to remove Vietnamese stopwords
        use_pos_tags (bool): Whether to use POS tagging
        
    Returns:
        str: Advanced processed text
    """
    # Basic preprocessing
    text = preprocess_text(text)
    
    if not text:
        return ""
    
    try:
        # Tokenize Vietnamese text
        tokens = word_tokenize(text)
        
        # POS tagging if requested
        if use_pos_tags:
            pos_tags = pos_tag(text)
            # Keep only nouns, adjectives, and verbs
            important_pos = ['N', 'A', 'V']
            tokens = [word for word, pos in pos_tags if any(pos.startswith(p) for p in important_pos)]
        
        # Remove stopwords if requested
        if remove_stopwords:
            vietnamese_stopwords = {
                'là', 'của', 'và', 'có', 'trong', 'được', 'một', 'với', 'cho', 'từ',
                'này', 'đó', 'các', 'để', 'khi', 'về', 'như', 'sau', 'tại', 'theo',
                'còn', 'cũng', 'đã', 'sẽ', 'bị', 'hay', 'hoặc', 'nếu', 'mà', 'thì',
                'rằng', 'những', 'nhiều', 'lại', 'nữa', 'chỉ', 'vào', 'ra', 'lên',
                'xuống', 'qua', 'đến', 'bên', 'dưới', 'trên', 'giữa', 'ngoài', 'trong'
            }
            tokens = [token for token in tokens if token.lower() not in vietnamese_stopwords]
        
        return ' '.join(tokens)
        
    except Exception as e:
        print(f"Error in advanced preprocessing: {e}")
        return text


def load_and_preprocess_data(file_path, text_column='processed', target_column=None):
    """
    Load and preprocess data from file
    
    Args:
        file_path (str): Path to data file
        text_column (str): Name of text column to preprocess
        target_column (str): Name of target column (optional)
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    try:
        # Load data based on file extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")
        
        # Check if text column exists
        if text_column not in df.columns:
            available_cols = [col for col in df.columns if df[col].dtype == 'object']
            if available_cols:
                text_column = available_cols[0]
                print(f"Text column not found, using: {text_column}")
            else:
                raise ValueError("No text column found")
        
        # Remove rows with missing text
        df = df.dropna(subset=[text_column])
        
        # Preprocess text
        df['processed_text'] = df[text_column].apply(preprocess_text)
        
        # Remove empty processed texts
        df = df[df['processed_text'].str.len() > 0]
        
        # Reset index
        df = df.reset_index(drop=True)
        
        print(f"Loaded and preprocessed {len(df)} records")
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def create_sentiment_labels(df, rating_column='Rating', positive_threshold=4, negative_threshold=2):
    """
    Create sentiment labels based on rating scores
    
    Args:
        df (pd.DataFrame): Input dataframe
        rating_column (str): Name of rating column
        positive_threshold (int): Minimum rating for positive sentiment
        negative_threshold (int): Maximum rating for negative sentiment
        
    Returns:
        pd.DataFrame: Dataframe with sentiment labels
    """
    if rating_column not in df.columns:
        print(f"Rating column '{rating_column}' not found")
        return df
    
    def rating_to_sentiment(rating):
        if pd.isna(rating):
            return 'neutral'
        if rating >= positive_threshold:
            return 'positive'
        elif rating <= negative_threshold:
            return 'negative'
        else:
            return 'neutral'
    
    df['sentiment'] = df[rating_column].apply(rating_to_sentiment)
    
    # Create numerical labels
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['sentiment_label'] = df['sentiment'].map(sentiment_map)
    
    return df


def balance_dataset(df, target_column='sentiment_label', method='undersample'):
    """
    Balance dataset for sentiment analysis
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of target column
        method (str): Balancing method ('undersample', 'oversample')
        
    Returns:
        pd.DataFrame: Balanced dataframe
    """
    if target_column not in df.columns:
        print(f"Target column '{target_column}' not found")
        return df
    
    class_counts = df[target_column].value_counts()
    print(f"Original class distribution:\n{class_counts}")
    
    if method == 'undersample':
        # Undersample to the smallest class
        min_count = class_counts.min()
        balanced_dfs = []
        
        for class_label in class_counts.index:
            class_df = df[df[target_column] == class_label]
            sampled_df = class_df.sample(n=min_count, random_state=42)
            balanced_dfs.append(sampled_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
    elif method == 'oversample':
        # Oversample to the largest class
        max_count = class_counts.max()
        balanced_dfs = []
        
        for class_label in class_counts.index:
            class_df = df[df[target_column] == class_label]
            if len(class_df) < max_count:
                # Oversample with replacement
                additional_samples = max_count - len(class_df)
                oversampled = class_df.sample(n=additional_samples, replace=True, random_state=42)
                class_df = pd.concat([class_df, oversampled], ignore_index=True)
            balanced_dfs.append(class_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    else:
        print("Unknown balancing method. Returning original dataframe.")
        return df
    
    # Shuffle the balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    balanced_counts = balanced_df[target_column].value_counts()
    print(f"Balanced class distribution:\n{balanced_counts}")
    
    return balanced_df


def extract_features(df, text_column='processed_text'):
    """
    Extract additional features from text
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of text column
        
    Returns:
        pd.DataFrame: Dataframe with additional features
    """
    if text_column not in df.columns:
        print(f"Text column '{text_column}' not found")
        return df
    
    # Text length features
    df['text_length'] = df[text_column].str.len()
    df['word_count'] = df[text_column].str.split().str.len()
    df['sentence_count'] = df[text_column].apply(lambda x: len(sent_tokenize(x)))
    df['avg_word_length'] = df[text_column].apply(
        lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
    )
    
    # Punctuation and special character features
    df['exclamation_count'] = df[text_column].str.count('!')
    df['question_count'] = df[text_column].str.count('?')
    df['uppercase_ratio'] = df[text_column].apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )
    
    # Sentiment indicators (basic)
    positive_words = ['tốt', 'hay', 'giỏi', 'xuất sắc', 'tuyệt vời', 'hài lòng', 'thích', 'yêu']
    negative_words = ['tệ', 'xấu', 'dở', 'không tốt', 'thất vọng', 'ghét', 'không thích']
    
    df['positive_word_count'] = df[text_column].apply(
        lambda x: sum(1 for word in positive_words if word in x.lower())
    )
    df['negative_word_count'] = df[text_column].apply(
        lambda x: sum(1 for word in negative_words if word in x.lower())
    )
    
    return df
