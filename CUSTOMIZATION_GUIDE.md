# CHASE-SQL Data Warehouse Customization Guide

## Overview

This technical guide shows how to adapt CHASE-SQL for data warehouse environments with large tables and minimal schema metadata. Focus is on hands-on implementation steps for practitioners working with analytical data.

## Key Differences: Data Warehouse vs Traditional OLTP

| Aspect | E-commerce DB (Current) | Data Warehouse (Target) |
|--------|------------------------|--------------------------|
| Schema | 5 small normalized tables | 10-100+ large denormalized tables |
| Relationships | Clear FK relationships | Implicit relationships, star/snowflake schema |
| Column Descriptions | Available | Often missing or incomplete |
| Query Patterns | CRUD operations | Analytical aggregations, time-series |
| Data Volume | 1K-10K rows per table | 1M-1B+ rows per table |

## Core Components to Modify

### 1. Database Schema Discovery (`database.py`)

Replace `ECommerceDB` with a data warehouse connector:

```python
# src/warehouse_db.py
import sqlalchemy
from sqlalchemy import MetaData, inspect
from typing import Dict, List, Any

class WarehouseDB:
    def __init__(self, connection_string: str):
        self.engine = sqlalchemy.create_engine(connection_string)
        self.connection_string = connection_string
        self.metadata = MetaData()
        self.inspector = inspect(self.engine)
        
    def get_schema_info(self) -> Dict[str, Any]:
        """Extract schema from data warehouse - no column descriptions needed"""
        tables = {}
        
        for table_name in self.inspector.get_table_names():
            columns = self.inspector.get_columns(table_name)
            
            tables[table_name] = {
                'columns': [col['name'] for col in columns],
                'types': [str(col['type']) for col in columns],
                # Skip descriptions - not available in most DWH
                'sample_values': self._get_sample_values(table_name, [col['name'] for col in columns])
            }
            
        return {
            'tables': tables,
            'table_count': len(tables),
            'relationships': self._infer_relationships()  # Best effort relationship detection
        }
    
    def _get_sample_values(self, table_name: str, columns: List[str], limit: int = 100) -> Dict[str, List]:
        """Get sample values for LSH matching - crucial for DWH without descriptions"""
        with self.engine.connect() as conn:
            # Get distinct values for string columns (for value matching)
            sample_values = {}
            
            for col in columns:
                try:
                    # Sample approach - get some distinct values
                    query = f"SELECT DISTINCT {col} FROM {table_name} WHERE {col} IS NOT NULL LIMIT {limit}"
                    result = conn.execute(sqlalchemy.text(query))
                    sample_values[col] = [str(row[0]) for row in result.fetchall()]
                except:
                    sample_values[col] = []  # Skip problematic columns
                    
        return sample_values
    
    def _infer_relationships(self) -> List[Dict[str, str]]:
        """Attempt to infer relationships from column names and foreign keys"""
        relationships = []
        
        # Get explicit foreign keys if available
        for table_name in self.inspector.get_table_names():
            fks = self.inspector.get_foreign_keys(table_name)
            for fk in fks:
                relationships.append({
                    'from_table': table_name,
                    'from_column': fk['constrained_columns'][0],
                    'to_table': fk['referred_table'], 
                    'to_column': fk['referred_columns'][0]
                })
        
        # Heuristic relationship inference for DWH
        # Look for common patterns: table_name_id, date columns, etc.
        tables = self.inspector.get_table_names()
        for table in tables:
            columns = [col['name'] for col in self.inspector.get_columns(table)]
            for col in columns:
                if col.endswith('_id') and col != 'id':
                    # Guess related table name
                    potential_table = col[:-3] + 's'  # user_id -> users
                    if potential_table in tables:
                        relationships.append({
                            'from_table': table,
                            'from_column': col,
                            'to_table': potential_table,
                            'to_column': 'id',
                            'inferred': True
                        })
        
        return relationships
    
    def execute_query(self, sql: str) -> Dict[str, Any]:
        """Execute query with timeout for large DWH tables"""
        try:
            with self.engine.connect() as conn:
                # Set query timeout
                conn = conn.execution_options(timeout=30)
                result = conn.execute(sqlalchemy.text(sql))
                rows = result.fetchall()
                
                return {
                    'success': True,
                    'results': [dict(row._mapping) for row in rows],
                    'row_count': len(rows),
                    'columns': list(rows[0]._mapping.keys()) if rows else []
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'results': [],
                'row_count': 0
            }
```

### 2. Enhanced Value Retrieval for DWH (`value_retrieval.py`)

Modify the value retrieval to work with sample values instead of full table scans:

```python
# src/value_retrieval.py modifications
class WarehouseValueRetrieval:
    def __init__(self, warehouse_db):
        self.db = warehouse_db
        self.schema_info = warehouse_db.get_schema_info()
        self.sample_values = self._build_value_index()
    
    def _build_value_index(self) -> Dict[str, List[str]]:
        """Build searchable index from sample values"""
        value_index = {}
        
        for table_name, table_info in self.schema_info['tables'].items():
            sample_values = table_info.get('sample_values', {})
            
            for column_name, values in sample_values.items():
                # Create searchable key
                key = f"{table_name}.{column_name}"
                value_index[key] = [str(v).lower() for v in values if v]
        
        return value_index
    
    def extract_keywords(self, question: str) -> List[str]:
        """Extract keywords from natural language question"""
        import re
        
        # Basic keyword extraction
        keywords = re.findall(r'\b[a-zA-Z]{2,}\b', question.lower())
        
        # Filter common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'over', 'after'}
        keywords = [k for k in keywords if k not in stop_words]
        
        return keywords
    
    def find_matching_values(self, keywords: List[str]) -> Dict[str, List[str]]:
        """Find matching values in sample data"""
        matches = {}
        
        for keyword in keywords:
            matches[keyword] = []
            
            # Search through sample values
            for table_col, values in self.sample_values.items():
                for value in values:
                    if keyword in value or value in keyword:
                        matches[keyword].append(f"{table_col} = '{value}'")
            
            # Limit matches to avoid overwhelming the system
            matches[keyword] = matches[keyword][:10]
        
        return matches
```

### 3. Data Warehouse Query Patterns (`knowledge_base.py`)

Replace e-commerce patterns with analytical query patterns:

```python
# src/warehouse_knowledge_base.py
ANALYTICAL_QUERY_PATTERNS = [
    {
        "description": "Time series aggregation",
        "sql_template": "SELECT DATE_TRUNC('{period}', date_column), {metric_function}({metric_column}) FROM {table_name} WHERE date_column >= '{start_date}' GROUP BY 1 ORDER BY 1",
        "keywords": ["trend", "over time", "monthly", "daily", "weekly"],
        "category": "time_series"
    },
    {
        "description": "Top N analysis", 
        "sql_template": "SELECT {dimension_column}, {metric_function}({metric_column}) as {metric_name} FROM {table_name} GROUP BY {dimension_column} ORDER BY {metric_name} DESC LIMIT {n}",
        "keywords": ["top", "highest", "best", "most"],
        "category": "ranking"
    },
    {
        "description": "Cohort analysis",
        "sql_template": "SELECT cohort_month, period_number, COUNT(*) as users FROM (SELECT user_id, DATE_TRUNC('month', first_date) as cohort_month, DATEDIFF('month', first_date, activity_date) as period_number FROM user_activity) GROUP BY 1, 2",
        "keywords": ["cohort", "retention", "lifecycle"],
        "category": "cohort"
    },
    {
        "description": "Funnel analysis",
        "sql_template": "SELECT step, COUNT(DISTINCT user_id) as users, LAG(COUNT(DISTINCT user_id)) OVER (ORDER BY step) as prev_step FROM funnel_events GROUP BY step ORDER BY step",
        "keywords": ["funnel", "conversion", "step", "drop-off"],
        "category": "funnel"
    }
]

class WarehouseKnowledgeBase:
    def __init__(self):
        self.patterns = ANALYTICAL_QUERY_PATTERNS
    
    def get_relevant_patterns(self, keywords: List[str]) -> List[Dict]:
        """Get patterns matching question keywords"""
        relevant = []
        
        for pattern in self.patterns:
            pattern_keywords = pattern['keywords']
            if any(keyword in ' '.join(keywords) for keyword in pattern_keywords):
                relevant.append(pattern)
        
        return relevant[:5]  # Limit to top 5 most relevant
```

### 4. Update Main System (`chase_sql.py`)

Modify the main system to use warehouse components:

```python
# src/chase_sql.py modifications
from warehouse_db import WarehouseDB
from warehouse_knowledge_base import WarehouseKnowledgeBase
from value_retrieval import WarehouseValueRetrieval

class ChaseSQL:
    def __init__(self, connection_string: str):
        # Use warehouse database
        self.db = WarehouseDB(connection_string)
        
        # Use warehouse-specific components
        self.knowledge_base = WarehouseKnowledgeBase()
        self.value_retrieval = WarehouseValueRetrieval(self.db)
        
        # Keep existing generators but update them to use warehouse schema
        self.generators = {
            'llm': LLMGenerator(self.db, self.knowledge_base),  # Recommended for DWH
            'divide_conquer': DivideConquerGenerator(self.db, self.knowledge_base),
            'query_plan': QueryPlanGenerator(self.db, self.knowledge_base)
        }
        
        self.query_fixer = QueryFixer(self.db)
        self.selection_agent = SelectionAgent(self.db)
        self.query_explainer = QueryExplainer()
```

## Database Connection Examples

### PostgreSQL Data Warehouse
```python
connection_string = "postgresql://user:password@warehouse-host:5432/analytics_db"
system = ChaseSQL(connection_string)
```

### Snowflake
```python
from sqlalchemy import create_engine
connection_string = "snowflake://user:password@account/database?warehouse=compute_wh&role=analyst"
system = ChaseSQL(connection_string)
```

### BigQuery (via SQLAlchemy)
```python
connection_string = "bigquery://project-id/dataset-id"
system = ChaseSQL(connection_string)
```

## Handling Large Tables

### 1. Query Performance Optimization

```python
# Add query limits and timeouts
class WarehouseDB:
    def execute_query(self, sql: str, limit: int = 1000) -> Dict[str, Any]:
        # Add LIMIT clause if not present
        if 'LIMIT' not in sql.upper():
            sql = f"{sql} LIMIT {limit}"
        
        # Add timeout
        with self.engine.connect() as conn:
            conn = conn.execution_options(timeout=60)  # 60 second timeout
            # ... rest of execution
```

### 2. Schema Sampling Strategy

```python
# Sample large tables for value discovery
def get_sample_values(self, table_name: str, sample_percent: float = 0.1):
    """Use TABLESAMPLE for large tables"""
    query = f"""
    SELECT column_name, column_value 
    FROM (
        SELECT * FROM {table_name} 
        TABLESAMPLE SYSTEM ({sample_percent})
    ) 
    UNPIVOT (column_value FOR column_name IN ({', '.join(columns)}))
    """
    # Execute sampling query
```

## Configuration Setup

### 1. Environment Variables
```bash
# .env file
DATABASE_CONNECTION_STRING=postgresql://user:pass@host:5432/warehouse
DATABASE_TYPE=postgresql
QUERY_TIMEOUT=60
SAMPLE_SIZE_LIMIT=1000
```

### 2. Config File
```python
# src/config.py additions
class Config:
    @staticmethod
    def get_warehouse_connection() -> str:
        return os.getenv('DATABASE_CONNECTION_STRING')
    
    @staticmethod
    def get_query_timeout() -> int:
        return int(os.getenv('QUERY_TIMEOUT', '60'))
    
    @staticmethod
    def get_sample_limit() -> int:
        return int(os.getenv('SAMPLE_SIZE_LIMIT', '1000'))
```

## Testing Your Setup

### 1. Basic Connection Test
```python
# test_warehouse_setup.py
def test_connection():
    db = WarehouseDB("your_connection_string")
    schema = db.get_schema_info()
    print(f"Found {len(schema['tables'])} tables")
    for table_name in list(schema['tables'].keys())[:5]:
        print(f"- {table_name}")

if __name__ == "__main__":
    test_connection()
```

### 2. Query Generation Test
```python
def test_query_generation():
    system = ChaseSQL("your_connection_string")
    
    test_questions = [
        "What are the top 10 values by revenue last month?",
        "Show me the trend of daily active users over the past year",
        "Which customer segments have the highest conversion rates?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        result = system.process_question(question)
        print(f"Generated SQL: {result['sql']}")
        print(f"Success: {result['execution_result']['success']}")
```

## Performance Tuning Tips

1. **Use LLM Generator**: Most effective for complex analytical queries
2. **Limit Sample Data**: Keep value samples under 1000 items per column
3. **Add Query Timeouts**: Prevent long-running queries from hanging
4. **Cache Schema Info**: Store schema metadata to avoid repeated discovery
5. **Use Query Limits**: Always add LIMIT clauses for exploration queries

## Common Issues and Solutions

| Problem | Solution |
|---------|----------|
| Schema discovery too slow | Use async discovery, cache results |
| Too many tables confuse system | Filter to most relevant tables only |
| No column descriptions | Rely on sample values and column names |
| Queries time out | Add aggressive LIMIT clauses |
| Poor value matching | Improve sample value collection |

This guide provides the technical foundation for adapting CHASE-SQL to data warehouse environments without extensive metadata.