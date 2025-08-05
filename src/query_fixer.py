"""
Query Fixer module implementing self-reflection based error correction.
Based on the CHASE-SQL paper's approach to fixing syntactically incorrect queries.
"""

import sqlite3
import re
from typing import Dict, Any, Optional, List

class QueryFixer:
    """
    Implements query fixing using self-reflection method.
    Fixes syntax errors and empty result issues in generated SQL queries.
    """
    
    def __init__(self, db_path: str, max_attempts: int = 3):
        self.db_path = db_path
        self.max_attempts = max_attempts
    
    def fix_query(self, query: str, question: str, error_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Attempt to fix a SQL query that produced an error or empty results.
        
        Args:
            query: The SQL query that needs fixing
            question: The original natural language question
            error_message: The error message if query failed to execute
            
        Returns:
            Dict containing fixed query and fixing details
        """
        
        fixing_attempts = []
        current_query = query
        
        for attempt in range(self.max_attempts):
            # Analyze the issue
            issue_type = self._analyze_issue(current_query, error_message)
            
            # Apply appropriate fix
            fixed_query = self._apply_fix(current_query, issue_type, question, error_message)
            
            # Test the fixed query
            test_result = self._test_query(fixed_query)
            
            fixing_attempts.append({
                'attempt': attempt + 1,
                'issue_identified': issue_type,
                'original_query': current_query,
                'fixed_query': fixed_query,
                'test_result': test_result
            })
            
            if test_result['success']:
                return {
                    'fixed_query': fixed_query,
                    'success': True,
                    'attempts': fixing_attempts,
                    'final_error': None
                }
            
            # Update for next iteration
            current_query = fixed_query
            error_message = test_result.get('error', 'Empty results')
        
        # All attempts failed
        return {
            'fixed_query': current_query,
            'success': False,
            'attempts': fixing_attempts,
            'final_error': error_message
        }
    
    def _analyze_issue(self, query: str, error_message: Optional[str]) -> str:
        """Analyze what type of issue the query has."""
        
        if error_message:
            error_lower = error_message.lower()
            
            # Common syntax errors
            if 'syntax error' in error_lower:
                if 'near' in error_lower:
                    return 'syntax_error_near'
                return 'syntax_error_general'
            
            elif 'no such column' in error_lower:
                return 'invalid_column'
            
            elif 'no such table' in error_lower:
                return 'invalid_table'
            
            elif 'ambiguous' in error_lower:
                return 'ambiguous_column'
            
            elif 'misuse of aggregate' in error_lower:
                return 'aggregate_misuse'
            
            else:
                return 'unknown_error'
        
        else:
            # Query executed but returned empty results
            return 'empty_results'
    
    def _apply_fix(self, query: str, issue_type: str, question: str, error_message: Optional[str]) -> str:
        """Apply appropriate fix based on issue type."""
        
        if issue_type == 'syntax_error_near':
            return self._fix_syntax_error_near(query, error_message)
        
        elif issue_type == 'syntax_error_general':
            return self._fix_general_syntax_error(query)
        
        elif issue_type == 'invalid_column':
            return self._fix_invalid_column(query, error_message)
        
        elif issue_type == 'invalid_table':
            return self._fix_invalid_table(query, error_message)
        
        elif issue_type == 'ambiguous_column':
            return self._fix_ambiguous_column(query, error_message)
        
        elif issue_type == 'aggregate_misuse':
            return self._fix_aggregate_misuse(query)
        
        elif issue_type == 'empty_results':
            return self._fix_empty_results(query, question)
        
        else:
            # Try general fixes
            return self._apply_general_fixes(query)
    
    def _fix_syntax_error_near(self, query: str, error_message: str) -> str:
        """Fix syntax errors that indicate location."""
        
        # Extract the location of error
        match = re.search(r'near "([^"]+)"', error_message)
        if match:
            error_location = match.group(1)
            
            # Common fixes for specific patterns
            if error_location == 'FROM':
                # Missing comma in SELECT clause
                query = re.sub(r'(\w+)\s+(\w+)\s+FROM', r'\1, \2 FROM', query)
            
            elif error_location in ['WHERE', 'AND', 'OR']:
                # Fix spacing issues
                query = re.sub(rf'\s*{error_location}\s*', f' {error_location} ', query)
            
            elif error_location == 'GROUP':
                # Ensure GROUP BY is together
                query = query.replace('GROUP', 'GROUP BY')
                query = query.replace('GROUP BY BY', 'GROUP BY')
        
        return query
    
    def _fix_general_syntax_error(self, query: str) -> str:
        """Apply general syntax fixes."""
        
        # Fix common syntax issues
        fixes = [
            # Fix missing spaces
            (r'(\w+)\(', r'\1 ('),
            (r'\)(\w+)', r') \1'),
            
            # Fix SELECT issues
            (r'SELECT(\w+)', r'SELECT \1'),
            (r'SELECT\s+,', 'SELECT'),
            
            # Fix WHERE clause
            (r'WHERE\s+AND', 'WHERE'),
            (r'WHERE\s+OR', 'WHERE'),
            
            # Fix quotes
            (r"''", "'"),
            (r'""', '"'),
            
            # Fix JOIN syntax
            (r'JOIN(\w+)', r'JOIN \1'),
            (r'ON(\w+)', r'ON \1'),
            
            # Remove trailing commas
            (r',\s*FROM', ' FROM'),
            (r',\s*WHERE', ' WHERE'),
            (r',\s*GROUP', ' GROUP'),
            (r',\s*ORDER', ' ORDER'),
        ]
        
        for pattern, replacement in fixes:
            query = re.sub(pattern, replacement, query)
        
        return query.strip()
    
    def _fix_invalid_column(self, query: str, error_message: str) -> str:
        """Fix invalid column references."""
        
        # Extract the invalid column name
        match = re.search(r'no such column: (\w+)', error_message)
        if match:
            invalid_column = match.group(1)
            
            # Try common column name variations
            variations = {
                'name': ['name', 'product_name', 'customer_name', 'title'],
                'price': ['price', 'unit_price', 'total_price', 'amount'],
                'date': ['date', 'order_date', 'created_date', 'registration_date'],
                'id': ['id', 'customer_id', 'product_id', 'order_id'],
                'count': ['quantity', 'stock_quantity', 'count']
            }
            
            # Find similar column that exists
            for base, variants in variations.items():
                if invalid_column.lower() in [v.lower() for v in variants]:
                    # Try each variant
                    for variant in variants:
                        test_query = query.replace(invalid_column, variant)
                        if self._test_query(test_query)['success']:
                            return test_query
        
        # If no fix found, try adding table prefix
        tables = self._extract_tables_from_query(query)
        if tables:
            return query.replace(invalid_column, f"{tables[0]}.{invalid_column}")
        
        return query
    
    def _fix_invalid_table(self, query: str, error_message: str) -> str:
        """Fix invalid table references."""
        
        # Extract the invalid table name
        match = re.search(r'no such table: (\w+)', error_message)
        if match:
            invalid_table = match.group(1)
            
            # Common table name corrections
            table_corrections = {
                'customer': 'customers',
                'product': 'products',
                'order': 'orders',
                'order_item': 'order_items',
                'review': 'reviews',
                'user': 'customers',
                'item': 'products'
            }
            
            if invalid_table.lower() in table_corrections:
                correct_table = table_corrections[invalid_table.lower()]
                return query.replace(invalid_table, correct_table)
        
        return query
    
    def _fix_ambiguous_column(self, query: str, error_message: str) -> str:
        """Fix ambiguous column references by adding table prefixes."""
        
        # Extract the ambiguous column
        match = re.search(r'ambiguous column name: (\w+)', error_message)
        if match:
            ambiguous_column = match.group(1)
            
            # Find tables in query
            tables = self._extract_tables_from_query(query)
            
            if tables:
                # Add table prefix to standalone column references
                # But avoid changing already prefixed columns
                pattern = rf'\b{ambiguous_column}\b'
                
                # Replace only if not already prefixed
                def replace_if_not_prefixed(match):
                    # Check if preceded by a dot (meaning it's already prefixed)
                    start = match.start()
                    if start > 0 and query[start-1] == '.':
                        return match.group()
                    return f"{tables[0]}.{ambiguous_column}"
                
                query = re.sub(pattern, replace_if_not_prefixed, query)
        
        return query
    
    def _fix_aggregate_misuse(self, query: str) -> str:
        """Fix misuse of aggregate functions."""
        
        # Common fixes for aggregate misuse
        
        # If using aggregate without GROUP BY, add GROUP BY
        if re.search(r'(COUNT|SUM|AVG|MAX|MIN)\s*\(', query) and 'GROUP BY' not in query.upper():
            # Find non-aggregate columns in SELECT
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE | re.DOTALL)
            if select_match:
                select_clause = select_match.group(1)
                
                # Find non-aggregate columns
                columns = []
                for part in select_clause.split(','):
                    part = part.strip()
                    if not re.search(r'(COUNT|SUM|AVG|MAX|MIN)\s*\(', part, re.IGNORECASE):
                        # Extract column name
                        col_match = re.search(r'(\w+\.?\w*)', part)
                        if col_match:
                            columns.append(col_match.group(1))
                
                if columns:
                    # Add GROUP BY clause
                    if 'WHERE' in query.upper():
                        query = re.sub(r'(WHERE.*?)(\s+ORDER|\s*$)', rf'\1 GROUP BY {", ".join(columns)}\2', query, flags=re.IGNORECASE)
                    else:
                        query = re.sub(r'(FROM.*?)(\s+ORDER|\s*$)', rf'\1 GROUP BY {", ".join(columns)}\2', query, flags=re.IGNORECASE)
        
        return query
    
    def _fix_empty_results(self, query: str, question: str) -> str:
        """Fix queries that return empty results."""
        
        # Try relaxing constraints
        
        # 1. Change = to LIKE with wildcards
        query = re.sub(r"=\s*'([^']+)'", r"LIKE '%\1%'", query)
        
        # 2. Remove some WHERE conditions (keep only the first one)
        if 'WHERE' in query.upper():
            where_match = re.search(r'WHERE\s+(.*?)(\s+GROUP|\s+ORDER|\s*$)', query, re.IGNORECASE | re.DOTALL)
            if where_match:
                conditions = where_match.group(1).split(' AND ')
                if len(conditions) > 1:
                    # Keep only first condition
                    query = query.replace(where_match.group(1), conditions[0])
        
        # 3. Remove LIMIT if it's too restrictive
        query = re.sub(r'\s+LIMIT\s+\d+', '', query, flags=re.IGNORECASE)
        
        return query
    
    def _apply_general_fixes(self, query: str) -> str:
        """Apply various general fixes."""
        
        # Collection of common fixes
        query = self._fix_general_syntax_error(query)
        
        # Ensure proper spacing
        query = ' '.join(query.split())
        
        # Fix common SQL keywords case
        sql_keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'ON', 'GROUP BY', 
                       'ORDER BY', 'LIMIT', 'AS', 'AND', 'OR', 'IN', 'NOT']
        
        for keyword in sql_keywords:
            # Replace case-insensitively but preserve the standard case
            query = re.sub(rf'\b{keyword}\b', keyword, query, flags=re.IGNORECASE)
        
        return query
    
    def _extract_tables_from_query(self, query: str) -> List[str]:
        """Extract table names from query."""
        tables = []
        
        # FROM clause
        from_match = re.search(r'FROM\s+(\w+)', query, re.IGNORECASE)
        if from_match:
            tables.append(from_match.group(1))
        
        # JOIN clauses
        join_matches = re.findall(r'JOIN\s+(\w+)', query, re.IGNORECASE)
        tables.extend(join_matches)
        
        return tables
    
    def _test_query(self, query: str) -> Dict[str, Any]:
        """Test if a query executes successfully."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(query)
            results = cursor.fetchall()
            conn.close()
            
            return {
                'success': True,
                'row_count': len(results),
                'empty': len(results) == 0
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }