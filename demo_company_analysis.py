#!/usr/bin/env python3
"""
Demo script showing key features of the Company Analysis functionality
"""
import pandas as pd
import os

def demonstrate_company_analysis():
    """Demonstrate the key features of the company analysis"""
    print("🎯 ITViec Company Analysis - Feature Demo")
    print("=" * 50)
    
    # Load sample data
    data_path = "data/final_data.xlsx"
    if os.path.exists(data_path):
        df = pd.read_excel(data_path)
    else:
        data_path = "data/reviews.csv"
        df = pd.read_csv(data_path)
    
    print(f"📊 Loaded {len(df)} reviews from dataset")
    print(f"🏢 Total companies: {df['Company Name'].nunique()}")
    
    # Show sample companies
    companies = df['Company Name'].value_counts().head(10)
    print("\n🔝 Top 10 Companies by Review Count:")
    for company, count in companies.items():
        print(f"   • {company}: {count} reviews")
    
    # Show what analysis each company gets
    sample_company = companies.index[0]
    company_df = df[df['Company Name'] == sample_company]
    
    print(f"\n🎯 Sample Analysis for '{sample_company}':")
    print(f"   📝 Total Reviews: {len(company_df)}")
    
    if 'Rating' in company_df.columns:
        avg_rating = company_df['Rating'].mean()
        print(f"   ⭐ Average Rating: {avg_rating:.2f}/5")
    
    if 'Recommend?' in company_df.columns:
        recommend_pct = (company_df['Recommend?'] == 'Yes').mean() * 100
        print(f"   👍 Recommendation Rate: {recommend_pct:.1f}%")
    
    # Show clustering potential
    text_columns = ['Title', 'What I liked', 'Suggestions for improvement']
    available_columns = [col for col in text_columns if col in company_df.columns]
    
    if available_columns:
        print(f"   📝 Text columns available for clustering: {', '.join(available_columns)}")
        
        # Count non-empty text entries
        non_empty_texts = 0
        for _, row in company_df.iterrows():
            combined_text = ' '.join([str(row.get(col, '')) for col in available_columns])
            if len(combined_text.strip()) > 10:
                non_empty_texts += 1
        
        print(f"   🔍 Reviews suitable for clustering: {non_empty_texts}")
        
        if non_empty_texts >= 3:
            print(f"   ✅ Sufficient data for clustering analysis")
        else:
            print(f"   ⚠️  Limited data for clustering (need at least 3 meaningful reviews)")
    
    print("\n🚀 Available Features in Company Analysis:")
    print("   1. 📊 Company Overview & Statistics")
    print("   2. 🔍 Intelligent Review Clustering")
    print("   3. 🏷️  Keyword Extraction & Word Clouds")
    print("   4. 📈 Performance Comparison with Other Companies")
    print("   5. 📅 Temporal Analysis (if date data available)")
    print("   6. 📝 Recent Reviews Showcase")
    print("   7. 🎯 Cluster-based Insights & Patterns")
    
    print("\n🌐 To use the Company Analysis:")
    print("   1. Run: streamlit run app.py")
    print("   2. Navigate to 'Company Analysis' in the sidebar")
    print("   3. Select any company from the dropdown")
    print("   4. Explore clustering, keywords, and comparisons!")

if __name__ == "__main__":
    demonstrate_company_analysis()
