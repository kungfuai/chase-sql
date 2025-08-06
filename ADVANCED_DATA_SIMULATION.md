# Advanced Data Simulation Strategy

This document outlines the strategy for creating enterprise-grade data warehouse mock scenarios to support analytics and driver analysis tasks using the CHASE-SQL text-to-SQL system.

## Overview

The goal is to simulate realistic data warehouse environments with clickstream data, multiple data sources, and complex business analytics queries that mirror real-world scenarios for driver analysis, attribution modeling, and performance analytics.

## 1. Multi-Tier Database Architecture

### Current Implementation
- **Simple Tier**: Existing e-commerce database (customers, products, orders)
- **Purpose**: Quick development, basic testing, debugging
- **Scale**: 10 customers, 15 products, 50 orders

### New Analytics Tier
- **Advanced Tier**: Data warehouse-style clickstream analytics database
- **Purpose**: Enterprise-grade testing, complex analytics, performance benchmarking
- **Scale**: 100K+ events, 12+ months of data, realistic distributions

### Configuration System
```python
# Database selection via configuration
DATABASE_MODE = "simple"  # or "analytics" or "both"
```

## 2. Clickstream Analytics Schema Design

### Core Fact Tables

#### `fact_clicks` (Date Partitioned)
```sql
CREATE TABLE fact_clicks (
    click_id BIGINT PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(32),
    timestamp DATETIME NOT NULL,
    query_vertical_id INT,
    merchant_domain_id INT,
    traffic_source_id INT,
    click_position INT,
    device_type VARCHAR(20),
    user_agent_category VARCHAR(50),
    referrer_domain VARCHAR(100),
    page_url TEXT,
    -- Partitioned by: DATE(timestamp)
    FOREIGN KEY (query_vertical_id) REFERENCES dim_query_vertical(id),
    FOREIGN KEY (merchant_domain_id) REFERENCES dim_merchants(id),
    FOREIGN KEY (traffic_source_id) REFERENCES dim_traffic_sources(id)
);
```

#### `fact_conversions`
```sql
CREATE TABLE fact_conversions (
    conversion_id BIGINT PRIMARY KEY,
    click_id BIGINT,
    session_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(32) NOT NULL,
    conversion_timestamp DATETIME NOT NULL,
    conversion_value DECIMAL(10,2),
    conversion_type VARCHAR(50), -- 'purchase', 'signup', 'lead'
    attribution_model VARCHAR(30), -- 'first_click', 'last_click', 'linear'
    time_to_conversion INT, -- seconds between click and conversion
    FOREIGN KEY (click_id) REFERENCES fact_clicks(click_id)
);
```

#### `fact_ad_spend`
```sql
CREATE TABLE fact_ad_spend (
    spend_id BIGINT PRIMARY KEY,
    date DATE NOT NULL,
    traffic_source_id INT,
    campaign_id VARCHAR(50),
    ad_group_id VARCHAR(50),
    keyword VARCHAR(200),
    spend_amount DECIMAL(12,2),
    impressions BIGINT,
    clicks BIGINT,
    FOREIGN KEY (traffic_source_id) REFERENCES dim_traffic_sources(id)
);
```

### Dimension Tables

#### `dim_query_vertical`
```sql
CREATE TABLE dim_query_vertical (
    id INT PRIMARY KEY,
    vertical_name VARCHAR(50), -- 'fashion', 'electronics', 'home', 'travel'
    vertical_category VARCHAR(30), -- 'product', 'service', 'information'
    parent_vertical_id INT,
    intent_score DECIMAL(3,2), -- 0.0-1.0 purchase intent probability
    avg_conversion_rate DECIMAL(5,4)
);
```

#### `dim_merchants`
```sql
CREATE TABLE dim_merchants (
    id INT PRIMARY KEY,
    domain VARCHAR(100) UNIQUE,
    merchant_name VARCHAR(200),
    category VARCHAR(50),
    tier VARCHAR(20), -- 'premium', 'mid_tier', 'budget'
    commission_rate DECIMAL(5,4),
    avg_rating DECIMAL(3,2),
    is_brand BOOLEAN
);
```

#### `dim_traffic_sources`
```sql
CREATE TABLE dim_traffic_sources (
    id INT PRIMARY KEY,
    source_name VARCHAR(50), -- 'google_organic', 'facebook_ads', 'affiliate'
    source_type VARCHAR(30), -- 'organic', 'paid', 'social', 'direct', 'referral'
    source_category VARCHAR(20), -- 'search', 'social', 'email', 'display'
    is_third_party BOOLEAN,
    data_quality_score DECIMAL(3,2) -- tracking accuracy score
);
```

#### `user_sessions`
```sql
CREATE TABLE user_sessions (
    session_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(32),
    session_start DATETIME,
    session_end DATETIME,
    total_clicks INT,
    total_conversions INT,
    session_value DECIMAL(10,2),
    device_type VARCHAR(20),
    geo_country VARCHAR(2),
    geo_region VARCHAR(50)
);
```

## 3. Business Analytics Query Files

### File Structure
```
/examples/analytics/
├── driver_analysis/
│   ├── conversion_drivers.sql
│   ├── revenue_attribution.sql
│   └── channel_performance.sql
├── cohort_analysis/
│   ├── user_retention.sql
│   └── ltv_analysis.sql
├── funnel_analysis/
│   ├── click_to_conversion.sql
│   └── multi_touch_attribution.sql
└── performance_optimization/
    ├── roi_by_vertical.sql
    └── cost_efficiency.sql
```

### Example Multi-Query Files

#### `conversion_drivers.sql`
```sql
-- Query 1: Top converting query verticals by month
WITH monthly_conversions AS (
    SELECT 
        DATE_TRUNC('month', c.conversion_timestamp) as month,
        qv.vertical_name,
        COUNT(*) as conversions,
        SUM(c.conversion_value) as revenue
    FROM fact_conversions c
    JOIN fact_clicks fc ON c.click_id = fc.click_id
    JOIN dim_query_vertical qv ON fc.query_vertical_id = qv.id
    WHERE c.conversion_timestamp >= DATE('now', '-12 months')
    GROUP BY 1, 2
)
SELECT 
    month,
    vertical_name,
    conversions,
    revenue,
    ROW_NUMBER() OVER (PARTITION BY month ORDER BY conversions DESC) as rank
FROM monthly_conversions;

-- Query 2: What drives higher conversion rates?
SELECT 
    qv.vertical_name,
    ts.source_type,
    COUNT(DISTINCT fc.click_id) as total_clicks,
    COUNT(DISTINCT c.conversion_id) as conversions,
    (COUNT(DISTINCT c.conversion_id) * 100.0 / COUNT(DISTINCT fc.click_id)) as conversion_rate,
    AVG(c.conversion_value) as avg_order_value
FROM fact_clicks fc
LEFT JOIN fact_conversions c ON fc.click_id = c.click_id
JOIN dim_query_vertical qv ON fc.query_vertical_id = qv.id
JOIN dim_traffic_sources ts ON fc.traffic_source_id = ts.id
WHERE fc.timestamp >= DATE('now', '-3 months')
GROUP BY 1, 2
HAVING total_clicks >= 100
ORDER BY conversion_rate DESC;

-- Query 3: Multi-touch attribution analysis
WITH user_journey AS (
    SELECT 
        c.user_id,
        c.conversion_id,
        fc.traffic_source_id,
        ts.source_name,
        ROW_NUMBER() OVER (PARTITION BY c.user_id, c.conversion_id ORDER BY fc.timestamp) as touch_sequence,
        COUNT(*) OVER (PARTITION BY c.user_id, c.conversion_id) as total_touches
    FROM fact_conversions c
    JOIN fact_clicks fc ON c.session_id = fc.session_id 
        AND fc.timestamp <= c.conversion_timestamp
    JOIN dim_traffic_sources ts ON fc.traffic_source_id = ts.id
    WHERE c.conversion_timestamp >= DATE('now', '-1 month')
)
SELECT 
    source_name,
    COUNT(CASE WHEN touch_sequence = 1 THEN 1 END) as first_touch_conversions,
    COUNT(CASE WHEN touch_sequence = total_touches THEN 1 END) as last_touch_conversions,
    COUNT(*) as total_touch_points,
    AVG(total_touches) as avg_touches_per_conversion
FROM user_journey
GROUP BY source_name
ORDER BY total_touch_points DESC;
```

### Query Categories and Examples

#### Driver Analysis Queries
- "What query verticals drive the highest conversion rates?"
- "Which traffic sources generate the most valuable customers?"
- "How does merchant tier impact conversion performance?"
- "What device types show the best ROI?"

#### Attribution Analysis
- "Multi-touch attribution: Which channels assist vs convert?"
- "Time-decay attribution modeling across touchpoints"
- "Cross-device attribution analysis"
- "Incrementality testing for paid vs organic traffic"

#### Performance Optimization
- "ROI analysis by traffic source and query vertical"
- "Cost per acquisition trends by campaign"
- "Lifetime value by acquisition channel"
- "Budget allocation optimization recommendations"

## 4. Database Functions Support

### Scalar Functions
```sql
-- Custom date functions
CREATE FUNCTION DATE_BUCKET(date_col DATE, bucket_size TEXT) 
RETURNS DATE AS 
BEGIN 
    -- Implementation for date bucketing
END;

-- Business logic functions
CREATE FUNCTION CALCULATE_LTV(user_id VARCHAR) 
RETURNS DECIMAL(10,2) AS
BEGIN
    -- Calculate customer lifetime value
END;
```

### Table-Valued Functions
```sql
-- Cohort analysis function
CREATE FUNCTION USER_COHORT_ANALYSIS(cohort_start DATE, cohort_end DATE)
RETURNS TABLE (
    cohort_month DATE,
    users_acquired INT,
    month_1_retention DECIMAL(5,4),
    month_3_retention DECIMAL(5,4),
    month_6_retention DECIMAL(5,4)
) AS
BEGIN
    -- Return cohort analysis results
END;

-- Funnel analysis function  
CREATE FUNCTION CONVERSION_FUNNEL(date_start DATE, date_end DATE)
RETURNS TABLE (
    funnel_step VARCHAR(50),
    users INT,
    conversion_rate DECIMAL(5,4),
    drop_off_rate DECIMAL(5,4)
) AS
BEGIN
    -- Return funnel analysis
END;
```

### Window Functions Support
- `ROW_NUMBER()`, `RANK()`, `DENSE_RANK()`
- `LAG()`, `LEAD()` for sequential analysis
- `FIRST_VALUE()`, `LAST_VALUE()`
- Moving averages: `AVG() OVER (ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)`

## 5. Realistic Data Generation

### Scale and Volume
- **fact_clicks**: 500K+ records over 12 months
- **fact_conversions**: 25K+ records (5% average conversion rate)
- **fact_ad_spend**: 10K+ daily spend records
- **user_sessions**: 100K+ unique sessions
- **Historical depth**: 18 months of data for trend analysis

### Realistic Distributions

#### Query Vertical Distribution (Zipfian)
```python
vertical_weights = {
    'fashion': 0.25,      # Most popular
    'electronics': 0.20,
    'home': 0.15,
    'travel': 0.12,
    'automotive': 0.08,
    'finance': 0.06,
    'health': 0.05,
    'other': 0.09
}
```

#### Conversion Rates by Vertical
```python
conversion_rates = {
    'finance': 0.12,      # Highest intent
    'automotive': 0.08,
    'electronics': 0.06,
    'fashion': 0.04,
    'travel': 0.03,       # Longer consideration
    'home': 0.05
}
```

#### Temporal Patterns
- **Seasonal trends**: Holiday shopping spikes, travel booking cycles
- **Weekly patterns**: Higher conversion rates on weekdays for B2B
- **Time-of-day**: Peak activity during business hours
- **Economic cycles**: Recession impact on luxury vs necessity categories

#### Traffic Source Mix
```python
traffic_distribution = {
    'google_organic': 0.35,
    'google_ads': 0.25,
    'facebook_ads': 0.15,
    'direct': 0.10,
    'affiliate': 0.08,
    'email': 0.04,
    'other': 0.03
}
```

### Data Relationships
- **Session clustering**: Multiple clicks per session with realistic patterns
- **User behavior**: Repeat visitors, cross-device tracking
- **Attribution chains**: Multi-touch customer journeys
- **Merchant performance**: Varying conversion rates by merchant tier

## 6. Enhanced Knowledge Base

### Analytics Query Patterns
```python
analytics_queries = [
    {
        "description": "Monthly conversion rate trend by traffic source",
        "sql": "SELECT DATE_TRUNC('month', fc.timestamp) as month, ts.source_name, (COUNT(c.conversion_id) * 100.0 / COUNT(fc.click_id)) as conversion_rate FROM fact_clicks fc LEFT JOIN fact_conversions c ON fc.click_id = c.click_id JOIN dim_traffic_sources ts ON fc.traffic_source_id = ts.id GROUP BY 1, 2",
        "category": "time_series_analysis",
        "complexity": "advanced"
    },
    {
        "description": "Customer lifetime value by acquisition channel",
        "sql": "WITH first_touch AS (SELECT DISTINCT user_id, FIRST_VALUE(ts.source_name) OVER (PARTITION BY fc.user_id ORDER BY fc.timestamp) as acquisition_channel FROM fact_clicks fc JOIN dim_traffic_sources ts ON fc.traffic_source_id = ts.id WHERE fc.user_id IS NOT NULL) SELECT ft.acquisition_channel, COUNT(DISTINCT ft.user_id) as customers, SUM(c.conversion_value) as total_revenue, AVG(c.conversion_value) as avg_ltv FROM first_touch ft JOIN fact_conversions c ON ft.user_id = c.user_id GROUP BY 1",
        "category": "cohort_analysis", 
        "complexity": "advanced"
    }
]
```

### Business Question Templates
- **Driver Analysis**: "What factors drive [metric] in [segment] during [timeframe]?"
- **Attribution**: "How do [channels] contribute to [outcome] for [customer_type]?"
- **Performance**: "What is the ROI of [campaign/channel] for [vertical] customers?"
- **Optimization**: "How should we reallocate [budget/resources] to maximize [objective]?"

## 7. Testing Framework Updates

### Analytics Test Categories
```python
analytics_test_categories = [
    {
        "category": "cohort_analysis",
        "queries": ["user retention by month", "LTV by acquisition channel"],
        "expected_features": ["WINDOW FUNCTIONS", "LAG/LEAD", "CTE"]
    },
    {
        "category": "funnel_analysis", 
        "queries": ["click to conversion funnel", "multi-step attribution"],
        "expected_features": ["SELF JOIN", "DATE ARITHMETIC", "CONDITIONAL AGGREGATION"]
    },
    {
        "category": "performance_analytics",
        "queries": ["ROI by traffic source", "cost efficiency analysis"],
        "expected_features": ["COMPLEX JOINS", "RATIO CALCULATIONS", "GROUPING SETS"]
    }
]
```

### Performance Benchmarks
- **Query Execution Time**: Target <2 seconds for complex analytics queries
- **Result Accuracy**: Compare against known ground truth for test scenarios
- **Scalability**: Performance with 100K+ row datasets
- **Memory Usage**: Monitor system resources during complex query generation

### Multi-Query File Processing
```python
def process_analytics_file(file_path: str) -> List[QueryResult]:
    """Process .sql files containing multiple queries separated by --;"""
    queries = parse_multi_query_file(file_path)
    results = []
    
    for query in queries:
        result = chase_sql.process_question(
            query.natural_language_description,
            expected_sql_pattern=query.sql_template
        )
        results.append(result)
    
    return results
```

## 8. Implementation Steps

### Phase 1: Core Infrastructure (Week 1-2)
1. **Create `AnalyticsDB` class** alongside existing `ECommerceDB`
   - Implement schema creation methods
   - Add date partitioning simulation
   - Create realistic data generators

2. **Database Function Support**
   - Add scalar function execution capability
   - Implement table-valued function framework
   - Add window function support validation

3. **Configuration System**
   - Add database mode selection
   - Environment-based configuration
   - Backward compatibility with simple dataset

### Phase 2: Data Generation (Week 2-3)
1. **Realistic Data Generators**
   - Implement Zipfian distributions for realistic patterns
   - Add temporal seasonality and trends
   - Create correlated data relationships (sessions, attribution chains)

2. **Large-Scale Data Creation**
   - Generate 500K+ click events over 18 months
   - Create realistic user journey patterns
   - Add merchant performance variations

### Phase 3: Query Examples and Knowledge Base (Week 3-4)
1. **Analytics Query Files**
   - Create `/examples/analytics/` directory structure
   - Implement multi-query file parsing
   - Add business question → SQL mappings

2. **Enhanced Knowledge Base**
   - Extend `QueryKnowledgeBase` with analytics patterns
   - Add complex query templates (CTEs, window functions)
   - Create business logic function examples

### Phase 4: Testing and Validation (Week 4-5)
1. **Analytics Test Suite**
   - Create performance benchmarks for complex queries
   - Add ground truth validation for known scenarios
   - Implement scalability tests

2. **Integration Testing**
   - Multi-query file processing validation
   - Cross-database compatibility testing
   - Function execution accuracy verification

### Phase 5: Documentation and Optimization (Week 5-6)
1. **Update Documentation**
   - Extend README.md with analytics capabilities
   - Create analytics query examples guide
   - Add performance tuning recommendations

2. **System Optimization**
   - Query generation performance improvements
   - Memory usage optimization for large datasets
   - Caching strategies for repeated analytics patterns

## Success Criteria

### Functional Requirements
- ✅ Support complex analytics queries (multi-table JOINs, CTEs, window functions)
- ✅ Process multi-query files with business context
- ✅ Execute database functions (scalar and table-valued)
- ✅ Handle realistic data warehouse scale (100K+ rows)

### Performance Requirements  
- ✅ Generate analytics SQL in <3 seconds for complex queries
- ✅ Execute queries on large datasets within reasonable time
- ✅ Support concurrent query processing
- ✅ Maintain accuracy rates >75% for advanced analytics queries

### Business Value
- ✅ Enable realistic driver analysis use cases
- ✅ Support multi-touch attribution modeling
- ✅ Provide ROI analysis capabilities
- ✅ Handle enterprise data warehouse query patterns

## Future Extensions

### Advanced Analytics Support
- **Machine Learning Integration**: Predictive models for conversion probability
- **Real-time Analytics**: Streaming data simulation
- **A/B Testing Framework**: Experiment analysis queries
- **Anomaly Detection**: Statistical outlier identification

### Enterprise Features
- **Multi-Database Support**: PostgreSQL, BigQuery, Snowflake dialects
- **Data Lineage**: Track data source and transformation history  
- **Governance**: Data quality metrics and validation rules
- **Security**: Row-level security and data masking simulation

This strategy provides a comprehensive foundation for creating enterprise-grade analytics dataset simulation that supports realistic business intelligence and driver analysis use cases while maintaining the educational and research value of the CHASE-SQL implementation.