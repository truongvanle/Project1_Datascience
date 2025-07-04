#!/usr/bin/env python3
"""
Simple test script to verify the company analysis functionality
"""
import pandas as pd
import os
import sys

def test_data_loading():
    """Test if data files can be loaded properly"""
    print("🔍 Testing data loading...")
    
    # Possible data paths
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
                
                if 'Company Name' in df.columns:
                    companies = df['Company Name'].nunique()
                    reviews = len(df)
                    print(f"✅ Found data file: {data_path}")
                    print(f"   📊 {reviews} reviews from {companies} companies")
                    print(f"   🏢 Sample companies: {df['Company Name'].unique()[:5].tolist()}")
                    return True
                    
            except Exception as e:
                print(f"❌ Error loading {data_path}: {e}")
                continue
    
    print("❌ No suitable data file found")
    return False

def test_imports():
    """Test if all required libraries can be imported"""
    print("🔍 Testing imports...")
    
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'sklearn',
        'matplotlib',
        'wordcloud'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'plotly':
                import plotly.express as px
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    return len(failed_imports) == 0

def main():
    """Run all tests"""
    print("🚀 ITViec Analytics Platform - Company Analysis Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    print()
    
    # Test data loading
    data_ok = test_data_loading()
    print()
    
    # Summary
    if imports_ok and data_ok:
        print("✅ All tests passed! The Company Analysis feature should work properly.")
        print("🚀 You can now run: streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the issues above.")
        if not imports_ok:
            print("   📦 Install missing packages with: pip install -r requirements.txt")
        if not data_ok:
            print("   📁 Ensure data files are available in the data/ directory")

if __name__ == "__main__":
    main()
