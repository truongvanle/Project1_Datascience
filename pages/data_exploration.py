import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page header
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
            padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
    <h1>📈 Data Exploration</h1>
    <p style="font-size: 1.2rem; margin-bottom: 0;">Comprehensive Analysis of ITViec Reviews Dataset</p>
</div>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    """Load the main dataset"""
    data_path = "data/reviews.csv"
    
    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    else:
        # Try alternative path
        alt_path = "Project1/final_data.xlsx"
        if os.path.exists(alt_path):
            try:
                df = pd.read_excel(alt_path)
                return df
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return None
        else:
            st.error("Data file not found")
            return None

# Load data
df = load_data()

if df is not None:
    st.success(f"✅ Successfully loaded {len(df)} reviews from {df['Company Name'].nunique() if 'Company Name' in df.columns else 'multiple'} companies")
    
    # Dataset overview
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dataset Overview", 
        "🏢 Company Analysis", 
        "⭐ Rating Analysis", 
        "📝 Text Analysis",
        "📅 Temporal Analysis"
    ])
    
    with tab1:
        st.markdown("### 📊 Dataset Overview")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", f"{len(df):,}")
        
        with col2:
            if 'Company Name' in df.columns:
                st.metric("Companies", f"{df['Company Name'].nunique():,}")
            else:
                st.metric("Companies", "N/A")
        
        with col3:
            if 'Rating' in df.columns:
                avg_rating = df['Rating'].mean()
                st.metric("Avg Rating", f"{avg_rating:.2f}")
            else:
                st.metric("Avg Rating", "N/A")
        
        with col4:
            # Calculate data completeness
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")
        
        # Data info
        st.markdown("#### 📋 Dataset Structure")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Column Information:**")
            info_data = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                non_null = df[col].count()
                null_count = df[col].isnull().sum()
                info_data.append({
                    'Column': col,
                    'Type': dtype,
                    'Non-Null': non_null,
                    'Null Count': null_count
                })
            
            info_df = pd.DataFrame(info_data)
            st.dataframe(info_df, use_container_width=True)
        
        with col2:
            st.markdown("**Missing Data Visualization:**")
            
            # Missing data heatmap
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                fig_missing = px.bar(
                    x=missing_data.values,
                    y=missing_data.index,
                    orientation='h',
                    title="Missing Values by Column"
                )
                fig_missing.update_layout(height=400)
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("🎉 No missing data found!")
        
        # Sample data
        st.markdown("#### 👀 Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Data types and statistics
        st.markdown("#### 📊 Statistical Summary")
        
        # Numerical columns summary
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            st.dataframe(df[numerical_cols].describe(), use_container_width=True)
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            st.markdown("**Categorical Columns Summary:**")
            cat_summary = []
            for col in categorical_cols[:5]:  # Show first 5 categorical columns
                unique_count = df[col].nunique()
                most_common = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
                cat_summary.append({
                    'Column': col,
                    'Unique Values': unique_count,
                    'Most Common': str(most_common)[:50] + "..." if len(str(most_common)) > 50 else str(most_common)
                })
            
            cat_df = pd.DataFrame(cat_summary)
            st.dataframe(cat_df, use_container_width=True)
    
    with tab2:
        st.markdown("### 🏢 Company Analysis")
        
        if 'Company Name' in df.columns:
            # Company distribution
            company_counts = df['Company Name'].value_counts().head(20)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Top 20 Companies by Review Count")
                fig_companies = px.bar(
                    x=company_counts.values,
                    y=company_counts.index,
                    orientation='h',
                    title="Companies with Most Reviews"
                )
                fig_companies.update_layout(height=600)
                st.plotly_chart(fig_companies, use_container_width=True)
            
            with col2:
                st.markdown("#### 🥧 Review Distribution")
                # Pie chart for top 10 companies
                top_10_companies = company_counts.head(10)
                others_count = company_counts.iloc[10:].sum()
                
                pie_data = top_10_companies.copy()
                if others_count > 0:
                    pie_data['Others'] = others_count
                
                fig_pie = px.pie(
                    values=pie_data.values,
                    names=pie_data.index,
                    title="Top 10 Companies + Others"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Company ratings analysis
            if 'Rating' in df.columns:
                st.markdown("#### ⭐ Company Ratings Analysis")
                
                # Average ratings by company (top 15)
                top_companies = df['Company Name'].value_counts().head(15).index
                company_ratings = df[df['Company Name'].isin(top_companies)].groupby('Company Name')['Rating'].agg(['mean', 'count', 'std']).reset_index()
                company_ratings = company_ratings.sort_values('mean', ascending=False)
                
                # Rating distribution plot
                fig_ratings = px.scatter(
                    company_ratings,
                    x='count',
                    y='mean',
                    size='count',
                    hover_data=['std'],
                    title="Company Ratings: Average vs Review Count",
                    labels={'count': 'Number of Reviews', 'mean': 'Average Rating'}
                )
                st.plotly_chart(fig_ratings, use_container_width=True)
                
                # Top rated companies (with minimum reviews)
                min_reviews = st.slider("Minimum reviews for rating analysis:", 5, 50, 10)
                qualified_companies = company_ratings[company_ratings['count'] >= min_reviews]
                
                if len(qualified_companies) > 0:
                    fig_top_rated = px.bar(
                        qualified_companies.head(10),
                        x='Company Name',
                        y='mean',
                        title=f"Top Rated Companies (Min {min_reviews} Reviews)",
                        labels={'mean': 'Average Rating'}
                    )
                    fig_top_rated.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_top_rated, use_container_width=True)
        else:
            st.warning("Company Name column not found in the dataset.")
    
    with tab3:
        st.markdown("### ⭐ Rating Analysis")
        
        if 'Rating' in df.columns:
            # Overall rating distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Overall Rating Distribution")
                
                rating_counts = df['Rating'].value_counts().sort_index()
                
                fig_rating_dist = px.bar(
                    x=rating_counts.index,
                    y=rating_counts.values,
                    title="Distribution of Ratings"
                )
                st.plotly_chart(fig_rating_dist, use_container_width=True)
                
                # Rating statistics
                rating_stats = df['Rating'].describe()
                st.markdown("**Rating Statistics:**")
                st.json({
                    "Mean": f"{rating_stats['mean']:.2f}",
                    "Median": f"{rating_stats['50%']:.2f}",
                    "Standard Deviation": f"{rating_stats['std']:.2f}",
                    "Min": f"{rating_stats['min']:.0f}",
                    "Max": f"{rating_stats['max']:.0f}"
                })
            
            with col2:
                st.markdown("#### 📈 Rating Trends")
                
                # Box plot
                fig_box = px.box(
                    df,
                    y='Rating',
                    title="Rating Distribution (Box Plot)"
                )
                st.plotly_chart(fig_box, use_container_width=True)
                
                # Histogram
                fig_hist = px.histogram(
                    df,
                    x='Rating',
                    nbins=10,
                    title="Rating Frequency Distribution"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Rating subcategories analysis
            rating_subcategories = [col for col in df.columns if 'Rating' in col or any(keyword in col.lower() for keyword in ['salary', 'training', 'management', 'culture', 'office'])]
            
            if len(rating_subcategories) > 1:
                st.markdown("#### 🎯 Rating Subcategories Analysis")
                
                # Correlation matrix
                rating_corr = df[rating_subcategories].corr()
                
                fig_corr = px.imshow(
                    rating_corr,
                    title="Rating Categories Correlation Matrix",
                    color_continuous_scale='RdBu'
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Average ratings by category
                avg_ratings = df[rating_subcategories].mean().sort_values(ascending=False)
                
                fig_avg = px.bar(
                    x=avg_ratings.values,
                    y=avg_ratings.index,
                    orientation='h',
                    title="Average Ratings by Category"
                )
                st.plotly_chart(fig_avg, use_container_width=True)
        else:
            st.warning("Rating column not found in the dataset.")
    
    with tab4:
        st.markdown("### 📝 Text Analysis")
        
        # Find text columns
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains text (not just categorical data)
                sample_text = str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else ""
                if len(sample_text.split()) > 3:  # Likely text if more than 3 words
                    text_columns.append(col)
        
        if text_columns:
            selected_text_col = st.selectbox("Select text column to analyze:", text_columns)
            
            if selected_text_col:
                # Text length analysis
                df['text_length'] = df[selected_text_col].astype(str).str.len()
                df['word_count'] = df[selected_text_col].astype(str).str.split().str.len()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📏 Text Length Distribution")
                    
                    fig_length = px.histogram(
                        df,
                        x='text_length',
                        nbins=50,
                        title="Character Count Distribution"
                    )
                    st.plotly_chart(fig_length, use_container_width=True)
                    
                    # Text length statistics
                    length_stats = df['text_length'].describe()
                    st.markdown("**Character Count Statistics:**")
                    st.json({
                        "Mean": f"{length_stats['mean']:.0f}",
                        "Median": f"{length_stats['50%']:.0f}",
                        "Max": f"{length_stats['max']:.0f}",
                        "Min": f"{length_stats['min']:.0f}"
                    })
                
                with col2:
                    st.markdown("#### 📊 Word Count Distribution")
                    
                    fig_words = px.histogram(
                        df,
                        x='word_count',
                        nbins=50,
                        title="Word Count Distribution"
                    )
                    st.plotly_chart(fig_words, use_container_width=True)
                    
                    # Word count statistics
                    word_stats = df['word_count'].describe()
                    st.markdown("**Word Count Statistics:**")
                    st.json({
                        "Mean": f"{word_stats['mean']:.0f}",
                        "Median": f"{word_stats['50%']:.0f}",
                        "Max": f"{word_stats['max']:.0f}",
                        "Min": f"{word_stats['min']:.0f}"
                    })
                
                # Sample texts
                st.markdown("#### 📖 Sample Texts")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Shortest Texts:**")
                    shortest = df.nsmallest(3, 'text_length')[selected_text_col]
                    for i, text in enumerate(shortest, 1):
                        st.write(f"{i}. {str(text)[:100]}...")
                
                with col2:
                    st.markdown("**Average Length Texts:**")
                    median_length = df['text_length'].median()
                    around_median = df[abs(df['text_length'] - median_length) < 10][selected_text_col].head(3)
                    for i, text in enumerate(around_median, 1):
                        st.write(f"{i}. {str(text)[:100]}...")
                
                with col3:
                    st.markdown("**Longest Texts:**")
                    longest = df.nlargest(3, 'text_length')[selected_text_col]
                    for i, text in enumerate(longest, 1):
                        st.write(f"{i}. {str(text)[:100]}...")
                
                # Text vs Rating correlation
                if 'Rating' in df.columns:
                    st.markdown("#### 🔗 Text Length vs Rating Correlation")
                    
                    fig_scatter = px.scatter(
                        df.sample(min(1000, len(df))),  # Sample for performance
                        x='text_length',
                        y='Rating',
                        opacity=0.6,
                        title="Text Length vs Rating",
                        trendline="ols"
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Correlation coefficient
                    correlation = df['text_length'].corr(df['Rating'])
                    st.info(f"Correlation between text length and rating: {correlation:.3f}")
        else:
            st.warning("No text columns found in the dataset.")
    
    with tab5:
        st.markdown("### 📅 Temporal Analysis")
        
        # Find date columns
        date_columns = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'day' in col.lower():
                date_columns.append(col)
        
        if date_columns:
            selected_date_col = st.selectbox("Select date column:", date_columns)
            
            if selected_date_col:
                try:
                    # Try to parse dates
                    df['parsed_date'] = pd.to_datetime(df[selected_date_col], errors='coerce')
                    
                    # Remove rows with invalid dates
                    valid_dates = df.dropna(subset=['parsed_date'])
                    
                    if len(valid_dates) > 0:
                        st.success(f"Successfully parsed {len(valid_dates)} dates")
                        
                        # Extract time components
                        valid_dates['year'] = valid_dates['parsed_date'].dt.year
                        valid_dates['month'] = valid_dates['parsed_date'].dt.month
                        valid_dates['day_of_week'] = valid_dates['parsed_date'].dt.day_name()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📈 Reviews Over Time")
                            
                            # Monthly trend
                            monthly_counts = valid_dates.groupby(valid_dates['parsed_date'].dt.to_period('M')).size()
                            
                            fig_timeline = px.line(
                                x=monthly_counts.index.astype(str),
                                y=monthly_counts.values,
                                title="Reviews per Month"
                            )
                            st.plotly_chart(fig_timeline, use_container_width=True)
                        
                        with col2:
                            st.markdown("#### 📊 Seasonal Patterns")
                            
                            # Day of week distribution
                            dow_counts = valid_dates['day_of_week'].value_counts()
                            
                            fig_dow = px.bar(
                                x=dow_counts.index,
                                y=dow_counts.values,
                                title="Reviews by Day of Week"
                            )
                            st.plotly_chart(fig_dow, use_container_width=True)
                        
                        # Yearly trend
                        if len(valid_dates['year'].unique()) > 1:
                            st.markdown("#### 📅 Yearly Trends")
                            
                            yearly_counts = valid_dates['year'].value_counts().sort_index()
                            
                            fig_yearly = px.bar(
                                x=yearly_counts.index,
                                y=yearly_counts.values,
                                title="Reviews per Year"
                            )
                            st.plotly_chart(fig_yearly, use_container_width=True)
                        
                        # Rating trends over time
                        if 'Rating' in df.columns:
                            st.markdown("#### ⭐ Rating Trends Over Time")
                            
                            monthly_ratings = valid_dates.groupby(valid_dates['parsed_date'].dt.to_period('M'))['Rating'].mean()
                            
                            fig_rating_trend = px.line(
                                x=monthly_ratings.index.astype(str),
                                y=monthly_ratings.values,
                                title="Average Rating Over Time"
                            )
                            st.plotly_chart(fig_rating_trend, use_container_width=True)
                    else:
                        st.warning("No valid dates found in the selected column.")
                        
                except Exception as e:
                    st.error(f"Error parsing dates: {e}")
        else:
            st.warning("No date columns found in the dataset.")
    
    # Download section
    st.markdown("---")
    st.markdown("## 📥 Download Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download full dataset
        csv_data = df.to_csv(index=False)
        st.download_button(
            "📊 Download Full Dataset",
            csv_data,
            f"itviec_reviews_full_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )
    
    with col2:
        # Download summary statistics
        if 'Rating' in df.columns:
            summary_stats = df.describe()
            summary_csv = summary_stats.to_csv()
            st.download_button(
                "📈 Download Summary Stats",
                summary_csv,
                f"summary_statistics_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
    
    with col3:
        # Download data info
        info_data = []
        for col in df.columns:
            info_data.append({
                'Column': col,
                'Type': str(df[col].dtype),
                'Non_Null_Count': df[col].count(),
                'Null_Count': df[col].isnull().sum(),
                'Unique_Values': df[col].nunique()
            })
        
        info_df = pd.DataFrame(info_data)
        info_csv = info_df.to_csv(index=False)
        st.download_button(
            "ℹ️ Download Data Info",
            info_csv,
            f"data_info_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )

else:
    st.error("❌ Unable to load data. Please check if the data files exist.")
    
    st.markdown("""
    ### Expected Data Locations:
    - `data/reviews.csv`
    - `Project1/final_data.xlsx`
    
    Please ensure one of these files exists and is accessible.
    """)
