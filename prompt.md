# Prompt for Exe 2

## Comprehensive Prompt for Clustering Analysis Improvement

**Objective**: Perform a complete overhaul of the clustering analysis notebook with enhanced EDA, separate analysis for different text columns, and comprehensive company-specific visualizations.

### Key Requirements:

#### 1. **Data Source Change**
- Use `Reviews.xlsx` from folder "Du lieu cung cap" instead of current data
- Replace all references to 'processed' column with separate analysis for:
  - `"What I liked"` column 
  - `"Suggestions for improvement"` column
- Perform clustering analysis on each column independently

#### 2. **Comprehensive EDA Requirements**
- Create beautiful, professional-looking visualizations
- Include the following EDA components:
  - Dataset overview (shape, missing values, data types)
  - Rating distribution analysis with statistical summary
  - Company-wise review count and rating analysis
  - Text length distribution for both columns
  - Word frequency analysis and word clouds for each column
  - Sentiment distribution visualization
  - Correlation analysis between different metrics
  - Time-based analysis if date columns exist

#### 3. **Visualization Standards**
- **CRITICAL**: All chart titles must NOT contain any symbols (no #, %, &, etc.)
- Use clean, professional titles in Vietnamese
- Implement consistent color schemes and styling
- Add proper axis labels and legends
- Use seaborn/matplotlib with professional styling

#### 4. **Enhanced Clustering Analysis**
- Apply clustering separately to "What I liked" and "Suggestions for improvement"
- For each text column, implement:
  - Count Vectorization with parameter tuning
  - LDA Topic Modeling (3-5 topics)
  - Multiple clustering algorithms (KMeans, Agglomerative, DBSCAN)
  - Optimal cluster number determination
  - Silhouette analysis and elbow method

#### 5. **Keyword Interpretation**
- For each identified cluster and topic:
  - Extract and display top 10-15 keywords
  - Provide detailed Vietnamese explanations of what each keyword cluster represents
  - Categorize keywords by themes (work environment, salary, management, etc.)
  - Create keyword importance visualizations

#### 6. **Implementation Approach**
- **Edit existing cells** rather than creating new ones where possible
- **Delete any error-prone or unnecessary cells** after implementation
- Maintain the existing notebook structure but enhance content
- Use the established functions but modify for dual-column analysis

#### 7. **Company-Specific Analysis Function**
Create a comprehensive function `analyze_single_company(company_name)` that generates:
- Company overview dashboard with key metrics
- Rating distribution for that company
- Word clouds for both "What I liked" and "Suggestions for improvement"
- Cluster assignment visualization for company reviews
- Topic distribution analysis
- Sentiment analysis breakdown
- Comparison with industry averages
- Strength and weakness summary based on clustering results

#### 8. **Enhanced Recommendation System**
- Modify recommendation functions to work with both text columns
- Weight recommendations based on both positive and negative feedback
- Include company size, industry, and other metadata in recommendations
- Create interactive filtering options

#### 9. **Code Quality Requirements**
- Add comprehensive docstrings for all functions
- Include error handling and validation
- Use type hints where appropriate
- Maintain consistent naming conventions
- Add progress bars for long-running operations

#### 10. **Final Deliverables**
- Clean, well-documented notebook with professional visualizations
- Working recommendation system with dual-column analysis
- Company-specific analysis function with rich visualizations
- Saved analysis results to Excel file
- Summary report with key insights and recommendations

### Implementation Instructions:
1. Start by updating data loading to use Reviews.xlsx
2. Create comprehensive EDA section with beautiful visualizations
3. Modify existing clustering pipeline for dual-column analysis
4. Enhance keyword extraction and interpretation
5. Implement company-specific analysis function
6. Update recommendation system
7. Clean up notebook by removing error cells
8. Test all functions and visualizations
9. Generate final summary and insights
10. Markdown cell for note taking or outline must write in Vietnamese.

**Note**: Prioritize editing existing cells over creating new ones. Ensure all visualizations follow the no-symbols-in-titles rule and provide meaningful insights in Vietnamese context.

