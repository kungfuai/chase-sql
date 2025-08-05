"""
Candidate generators implementing the three approaches from CHASE-SQL:
1. Divide and Conquer CoT
2. Query Plan CoT  
3. Online Synthetic Example Generation
"""

import re
import random
import sqlite3
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from knowledge_base import QueryKnowledgeBase
from value_retrieval import ValueRetrieval

class BaseCandidateGenerator(ABC):
    """Base class for all candidate generators."""
    
    def __init__(self, db_path: str, knowledge_base: QueryKnowledgeBase):
        self.db_path = db_path
        self.knowledge_base = knowledge_base
        self.value_retrieval = ValueRetrieval(db_path)
    
    @abstractmethod
    def generate_candidate(self, question: str, schema: str) -> Dict[str, Any]:
        """Generate a SQL candidate for the given question."""
        pass
    
    def get_schema_info(self) -> str:
        """Get database schema information."""
        conn = sqlite3.connect(self.db_path)
        schema_info = []
        
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        
        for table in tables:
            table_name = table[0]
            schema_info.append(f"\nTable: {table_name}")
            
            columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            for col in columns:
                col_name, col_type = col[1], col[2]
                schema_info.append(f"  {col_name}: {col_type}")
        
        conn.close()
        return "\n".join(schema_info)

class DivideConquerGenerator(BaseCandidateGenerator):
    """
    Divide and Conquer Chain-of-Thought generator.
    Breaks down complex queries into manageable sub-problems.
    """
    
    def generate_candidate(self, question: str, schema: str) -> Dict[str, Any]:
        """Generate SQL using divide and conquer approach."""
        
        # Step 1: Analyze the question and break it down
        analysis = self._analyze_question(question)
        
        # Step 2: Decompose into sub-questions
        sub_questions = self._decompose_question(question, analysis)
        
        # Step 3: Generate pseudo-SQL for each sub-question
        sub_sqls = []
        for sub_q in sub_questions:
            sub_sql = self._generate_sub_sql(sub_q, schema)
            sub_sqls.append({
                'sub_question': sub_q,
                'pseudo_sql': sub_sql
            })
        
        # Step 4: Combine into final SQL
        final_sql = self._combine_sub_sqls(question, sub_sqls, schema)
        
        # Step 5: Optimize the query
        optimized_sql = self._optimize_query(final_sql)
        
        return {
            'sql': optimized_sql,
            'reasoning': {
                'approach': 'divide_and_conquer',
                'analysis': analysis,
                'sub_questions': sub_questions,
                'sub_sqls': sub_sqls,
                'combination_step': final_sql,
                'final_optimized': optimized_sql
            },
            'confidence': self._calculate_confidence(question, optimized_sql)
        }
    
    def _analyze_question(self, question: str) -> Dict[str, Any]:
        """Analyze the question to understand its components."""
        analysis = {
            'main_intent': self._identify_main_intent(question),
            'entities': self._extract_entities(question),
            'constraints': self._extract_constraints(question),
            'aggregations': self._identify_aggregations(question),
            'comparisons': self._identify_comparisons(question)
        }
        return analysis
    
    def _identify_main_intent(self, question: str) -> str:
        """Identify the main intent of the question."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['how many', 'count', 'number of']):
            return 'count'
        elif any(word in question_lower for word in ['what is', 'show', 'list', 'get']):
            return 'retrieve'
        elif any(word in question_lower for word in ['average', 'avg', 'mean']):
            return 'average'
        elif any(word in question_lower for word in ['sum', 'total', 'amount']):
            return 'sum'
        elif any(word in question_lower for word in ['maximum', 'max', 'highest', 'most']):
            return 'maximum'
        elif any(word in question_lower for word in ['minimum', 'min', 'lowest', 'least']):
            return 'minimum'
        else:
            return 'retrieve'  # Default
    
    def _extract_entities(self, question: str) -> List[str]:
        """Extract potential database entities (tables/columns) from question."""
        entities = []
        
        # Common database entities in our e-commerce schema
        entity_mapping = {
            'customer': 'customers',
            'customers': 'customers',
            'user': 'customers',
            'users': 'customers',
            'product': 'products',
            'products': 'products', 
            'item': 'products',
            'items': 'products',
            'order': 'orders',
            'orders': 'orders',
            'purchase': 'orders',
            'purchases': 'orders',
            'review': 'reviews',
            'reviews': 'reviews',
            'rating': 'reviews',
            'ratings': 'reviews'
        }
        
        question_lower = question.lower()
        for keyword, table in entity_mapping.items():
            if keyword in question_lower:
                entities.append(table)
        
        return list(set(entities))  # Remove duplicates
    
    def _extract_constraints(self, question: str) -> List[str]:
        """Extract constraints/filters from the question."""
        constraints = []
        
        # Look for value-based constraints using value retrieval
        relevant_values = self.value_retrieval.retrieve_relevant_values(question)
        for keyword, matches in relevant_values.items():
            for match in matches[:1]:  # Top match per keyword
                constraint = f"{match['table']}.{match['column']} = '{match['value']}'"
                constraints.append(constraint)
        
        return constraints
    
    def _identify_aggregations(self, question: str) -> List[str]:
        """Identify aggregation functions needed."""
        question_lower = question.lower()
        aggregations = []
        
        if any(word in question_lower for word in ['how many', 'count', 'number of']):
            aggregations.append('COUNT')
        if any(word in question_lower for word in ['average', 'avg', 'mean']):
            aggregations.append('AVG')
        if any(word in question_lower for word in ['sum', 'total']):
            aggregations.append('SUM')
        if any(word in question_lower for word in ['maximum', 'max', 'highest']):
            aggregations.append('MAX')
        if any(word in question_lower for word in ['minimum', 'min', 'lowest']):
            aggregations.append('MIN')
        
        return aggregations
    
    def _identify_comparisons(self, question: str) -> List[str]:
        """Identify comparison operations."""
        question_lower = question.lower()
        comparisons = []
        
        if any(word in question_lower for word in ['more than', 'greater than', 'above']):
            comparisons.append('>')
        if any(word in question_lower for word in ['less than', 'below', 'under']):
            comparisons.append('<')
        if any(word in question_lower for word in ['at least', 'minimum']):
            comparisons.append('>=')
        if any(word in question_lower for word in ['at most', 'maximum']):
            comparisons.append('<=')
        
        return comparisons
    
    def _decompose_question(self, question: str, analysis: Dict) -> List[str]:
        """Break down the main question into sub-questions."""
        sub_questions = []
        
        # Based on the analysis, create logical sub-questions
        if analysis['main_intent'] == 'count':
            if analysis['constraints']:
                sub_questions.append("What are the relevant filtering conditions?")
                sub_questions.append("Which records match these conditions?")
                sub_questions.append("How many matching records are there?")
            else:
                sub_questions.append("Count all records in the target table")
        
        elif analysis['main_intent'] in ['maximum', 'minimum']:
            sub_questions.append("Which table contains the target values?")
            sub_questions.append("What column should be compared?")
            sub_questions.append("Apply the aggregation function")
        
        elif len(analysis['entities']) > 1:
            sub_questions.append("Which tables need to be joined?")
            sub_questions.append("What are the join conditions?")
            sub_questions.append("Apply filtering and aggregation")
        
        else:
            sub_questions.append("Identify target table and columns")
            sub_questions.append("Apply any filtering conditions")
            sub_questions.append("Format the final result")
        
        return sub_questions
    
    def _generate_sub_sql(self, sub_question: str, schema: str) -> str:
        """Generate pseudo-SQL for a sub-question."""
        # This is a simplified version - in a real implementation,
        # this would involve more sophisticated NL-to-SQL conversion
        
        if "filtering conditions" in sub_question.lower():
            constraints = self.value_retrieval.suggest_where_clauses("")
            return f"WHERE {' AND '.join(constraints[:2])}" if constraints else "WHERE 1=1"
        
        elif "join" in sub_question.lower():
            return "JOIN table2 ON table1.id = table2.foreign_key_id"
        
        elif "count" in sub_question.lower():
            return "SELECT COUNT(*) FROM target_table"
        
        else:
            return "SELECT columns FROM table"
    
    def _combine_sub_sqls(self, question: str, sub_sqls: List[Dict], schema: str) -> str:
        """Combine sub-SQLs into a complete query."""
        # Look for similar questions in knowledge base
        similar_queries = self.knowledge_base.search_queries(question.split()[:3])
        
        if similar_queries:
            # Use the most similar query as a template
            template = similar_queries[0]['sql']
            return self._adapt_template(template, question)
        
        # Fallback: construct basic query structure
        return self._construct_basic_query(question, schema)
    
    def _adapt_template(self, template: str, question: str) -> str:
        """Adapt a template query to the current question."""
        # Simple adaptation - replace values with retrieved ones
        relevant_values = self.value_retrieval.retrieve_relevant_values(question)
        
        adapted = template
        for keyword, matches in relevant_values.items():
            if matches:
                value = matches[0]['value']
                # Simple replacement strategy
                adapted = adapted.replace("'placeholder'", f"'{value}'")
        
        return adapted
    
    def _construct_basic_query(self, question: str, schema: str) -> str:
        """Construct a basic query when no template is available."""
        analysis = self._analyze_question(question)
        
        # Basic query construction
        if analysis['main_intent'] == 'count':
            if analysis['entities']:
                table = analysis['entities'][0]
                return f"SELECT COUNT(*) FROM {table}"
            else:
                return "SELECT COUNT(*) FROM customers"  # Default table
        
        elif analysis['entities']:
            table = analysis['entities'][0]
            return f"SELECT * FROM {table} LIMIT 10"
        
        else:
            return "SELECT COUNT(*) FROM customers"
    
    def _optimize_query(self, query: str) -> str:
        """Apply basic query optimizations."""
        # Remove redundant conditions
        optimized = query
        
        # Remove duplicate WHERE clauses
        if optimized.count('WHERE') > 1:
            parts = optimized.split('WHERE')
            if len(parts) > 2:
                conditions = []
                for part in parts[1:]:
                    conditions.append(part.split()[0])  # First condition
                optimized = parts[0] + 'WHERE ' + ' AND '.join(conditions)
        
        return optimized.strip()
    
    def _calculate_confidence(self, question: str, sql: str) -> float:
        """Calculate confidence score for the generated SQL."""
        # Simple confidence calculation based on query complexity and matches
        base_confidence = 0.7
        
        # Boost confidence if we found good value matches
        relevant_values = self.value_retrieval.retrieve_relevant_values(question)
        if relevant_values:
            base_confidence += 0.1
        
        # Boost confidence if query uses common patterns
        if any(pattern in sql.upper() for pattern in ['JOIN', 'GROUP BY', 'COUNT']):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)


class QueryPlanGenerator(BaseCandidateGenerator):
    """
    Query Plan Chain-of-Thought generator.
    Mirrors database execution steps to generate SQL queries.
    """
    
    def generate_candidate(self, question: str, schema: str) -> Dict[str, Any]:
        """Generate SQL using query execution plan approach."""
        
        # Step 1: Create query execution plan
        execution_plan = self._create_execution_plan(question, schema)
        
        # Step 2: Convert plan to SQL
        sql_query = self._plan_to_sql(execution_plan, question)
        
        return {
            'sql': sql_query,
            'reasoning': {
                'approach': 'query_plan',
                'execution_plan': execution_plan,
                'plan_steps': execution_plan['steps']
            },
            'confidence': self._calculate_plan_confidence(execution_plan, sql_query)
        }
    
    def _create_execution_plan(self, question: str, schema: str) -> Dict[str, Any]:
        """Create a human-readable query execution plan."""
        
        plan = {
            'steps': [],
            'tables_involved': [],
            'operations': [],
            'filters': [],
            'aggregations': []
        }
        
        # Step 1: Identify tables to scan
        entities = self._identify_required_tables(question)
        plan['tables_involved'] = entities
        plan['steps'].append(f"1. Scan tables: {', '.join(entities)}")
        
        # Step 2: Identify join operations
        if len(entities) > 1:
            joins = self._plan_joins(entities)
            plan['operations'].extend(joins)
            for i, join in enumerate(joins):
                plan['steps'].append(f"{i+2}. {join}")
        
        # Step 3: Apply filters
        filters = self._plan_filters(question)
        plan['filters'] = filters
        if filters:
            filter_step = len(plan['steps']) + 1
            plan['steps'].append(f"{filter_step}. Apply filters: {', '.join(filters)}")
        
        # Step 4: Apply aggregations
        aggregations = self._plan_aggregations(question)
        plan['aggregations'] = aggregations
        if aggregations:
            agg_step = len(plan['steps']) + 1
            plan['steps'].append(f"{agg_step}. Apply aggregation: {', '.join(aggregations)}")
        
        # Step 5: Format output
        output_step = len(plan['steps']) + 1
        plan['steps'].append(f"{output_step}. Format and return results")
        
        return plan
    
    def _identify_required_tables(self, question: str) -> List[str]:
        """Identify which tables are needed for the query."""
        question_lower = question.lower()
        tables = []
        
        # Map question keywords to tables
        table_keywords = {
            'customers': ['customer', 'customers', 'user', 'users', 'buyer', 'client'],
            'products': ['product', 'products', 'item', 'items', 'merchandise'],
            'orders': ['order', 'orders', 'purchase', 'purchases', 'transaction'],
            'order_items': ['order item', 'order items', 'purchased item'],
            'reviews': ['review', 'reviews', 'rating', 'ratings', 'feedback']
        }
        
        for table, keywords in table_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                tables.append(table)
        
        # If no specific tables identified, default based on question type
        if not tables:
            if any(word in question_lower for word in ['how many', 'count']):
                tables = ['customers']  # Most common count questions
            else:
                tables = ['products']  # Default to products
        
        return tables
    
    def _plan_joins(self, tables: List[str]) -> List[str]:
        """Plan the join operations between tables."""
        joins = []
        
        # Define common join patterns
        join_patterns = {
            ('customers', 'orders'): "JOIN orders ON customers.id = orders.customer_id",
            ('orders', 'order_items'): "JOIN order_items ON orders.id = order_items.order_id", 
            ('products', 'order_items'): "JOIN order_items ON products.id = order_items.product_id",
            ('products', 'reviews'): "JOIN reviews ON products.id = reviews.product_id",
            ('customers', 'reviews'): "JOIN reviews ON customers.id = reviews.customer_id"
        }
        
        # Generate joins for table pairs
        for i in range(len(tables)):
            for j in range(i+1, len(tables)):
                table1, table2 = tables[i], tables[j]
                
                # Check both directions
                if (table1, table2) in join_patterns:
                    joins.append(join_patterns[(table1, table2)])
                elif (table2, table1) in join_patterns:
                    joins.append(join_patterns[(table2, table1)])
        
        return joins
    
    def _plan_filters(self, question: str) -> List[str]:
        """Plan filtering operations based on the question."""
        filters = []
        
        # Use value retrieval to find potential filter conditions
        relevant_values = self.value_retrieval.retrieve_relevant_values(question)
        
        for keyword, matches in relevant_values.items():
            for match in matches[:1]:  # Top match per keyword
                filter_condition = f"{match['table']}.{match['column']} = '{match['value']}'"
                filters.append(filter_condition)
        
        # Add common filter patterns based on question
        question_lower = question.lower()
        
        if 'this year' in question_lower or '2023' in question_lower:
            filters.append("strftime('%Y', date_column) = '2023'")
        
        if 'last month' in question_lower:
            filters.append("date_column >= date('now', 'start of month', '-1 month')")
        
        return filters
    
    def _plan_aggregations(self, question: str) -> List[str]:
        """Plan aggregation operations."""
        question_lower = question.lower()
        aggregations = []
        
        if any(word in question_lower for word in ['how many', 'count', 'number of']):
            aggregations.append("COUNT(*)")
        elif any(word in question_lower for word in ['average', 'avg', 'mean']):
            aggregations.append("AVG(column)")
        elif any(word in question_lower for word in ['sum', 'total']):
            aggregations.append("SUM(column)")
        elif any(word in question_lower for word in ['maximum', 'max', 'highest']):
            aggregations.append("MAX(column)")
        elif any(word in question_lower for word in ['minimum', 'min', 'lowest']):
            aggregations.append("MIN(column)")
        
        return aggregations
    
    def _plan_to_sql(self, plan: Dict[str, Any], question: str) -> str:
        """Convert execution plan to actual SQL query."""
        
        # Start with base query structure
        if plan['aggregations']:
            if 'COUNT' in plan['aggregations'][0]:
                select_clause = "SELECT COUNT(*)"
            else:
                select_clause = f"SELECT {plan['aggregations'][0]}"
        else:
            select_clause = "SELECT *"
        
        # FROM clause
        if plan['tables_involved']:
            from_clause = f"FROM {plan['tables_involved'][0]}"
        else:
            from_clause = "FROM customers"
        
        # Add JOINs
        join_clauses = []
        for operation in plan['operations']:
            if 'JOIN' in operation:
                join_clauses.append(operation)
        
        # WHERE clause
        where_clause = ""
        if plan['filters']:
            where_clause = f"WHERE {' AND '.join(plan['filters'][:2])}"  # Limit to 2 filters
        
        # Combine all parts
        query_parts = [select_clause, from_clause] + join_clauses
        if where_clause:
            query_parts.append(where_clause)
        
        # Add ORDER BY for certain queries
        question_lower = question.lower()
        if any(word in question_lower for word in ['top', 'highest', 'best']):
            if 'price' in question_lower:
                query_parts.append("ORDER BY price DESC")
            elif 'rating' in question_lower:
                query_parts.append("ORDER BY rating DESC")
            query_parts.append("LIMIT 5")
        
        return " ".join(query_parts)
    
    def _calculate_plan_confidence(self, plan: Dict[str, Any], sql: str) -> float:
        """Calculate confidence based on plan completeness."""
        confidence = 0.8  # Base confidence for query plan approach
        
        # Boost for complete plans
        if plan['tables_involved'] and plan['steps']:
            confidence += 0.1
        
        # Boost for proper joins
        if len(plan['tables_involved']) > 1 and plan['operations']:
            confidence += 0.05
        
        return min(confidence, 1.0)


class OnlineSyntheticGenerator(BaseCandidateGenerator):
    """
    Online Synthetic Example Generation.
    Generates relevant few-shot examples on-the-fly to guide SQL generation.
    """
    
    def generate_candidate(self, question: str, schema: str) -> Dict[str, Any]:
        """Generate SQL using synthetic examples approach."""
        
        # Step 1: Generate relevant synthetic examples
        synthetic_examples = self._generate_synthetic_examples(question, schema)
        
        # Step 2: Use examples to guide SQL generation
        sql_query = self._generate_from_examples(question, synthetic_examples)
        
        return {
            'sql': sql_query,
            'reasoning': {
                'approach': 'synthetic_examples',
                'generated_examples': synthetic_examples,
                'example_based_reasoning': f"Generated {len(synthetic_examples)} relevant examples"
            },
            'confidence': self._calculate_synthetic_confidence(synthetic_examples, sql_query)
        }
    
    def _generate_synthetic_examples(self, question: str, schema: str, num_examples: int = 3) -> List[Dict]:
        """Generate relevant examples based on question characteristics."""
        
        examples = []
        
        # Get base examples from knowledge base
        question_keywords = question.lower().split()
        similar_queries = self.knowledge_base.search_queries(question_keywords[:3])
        
        if similar_queries:
            # Use similar queries as templates
            for query in similar_queries[:num_examples]:
                example = {
                    'question': query['description'],
                    'sql': query['sql'],
                    'category': query['category'],
                    'relevance': 'knowledge_base_match'
                }
                examples.append(example)
        
        # Fill remaining slots with generated examples
        while len(examples) < num_examples:
            synthetic_example = self._create_synthetic_example(question, schema)
            examples.append(synthetic_example)
        
        return examples[:num_examples]
    
    def _create_synthetic_example(self, question: str, schema: str) -> Dict:
        """Create a synthetic example based on question pattern."""
        
        # Analyze question to determine example type needed
        question_lower = question.lower()
        
        if 'count' in question_lower or 'how many' in question_lower:
            return {
                'question': "How many products are in the Electronics category?",
                'sql': "SELECT COUNT(*) FROM products WHERE category = 'Electronics'",
                'category': 'count_with_filter',
                'relevance': 'pattern_match'
            }
        
        elif 'average' in question_lower or 'avg' in question_lower:
            return {
                'question': "What is the average price of products?",
                'sql': "SELECT AVG(price) FROM products",
                'category': 'aggregation',
                'relevance': 'pattern_match'
            }
        
        elif any(word in question_lower for word in ['top', 'highest', 'best']):
            return {
                'question': "Which are the top 3 most expensive products?",
                'sql': "SELECT name, price FROM products ORDER BY price DESC LIMIT 3",
                'category': 'top_n',
                'relevance': 'pattern_match'
            }
        
        elif any(word in question_lower for word in ['join', 'customer', 'order']):
            return {
                'question': "Which customers have placed orders?",
                'sql': "SELECT DISTINCT c.name FROM customers c JOIN orders o ON c.id = o.customer_id",
                'category': 'join',
                'relevance': 'pattern_match'
            }
        
        else:
            # Default example
            return {
                'question': "Show all products",
                'sql': "SELECT * FROM products LIMIT 10",
                'category': 'simple_select',
                'relevance': 'default'
            }
    
    def _generate_from_examples(self, question: str, examples: List[Dict]) -> str:
        """Generate SQL query based on synthetic examples."""
        
        # Find the most relevant example
        best_example = self._find_best_example(question, examples)
        
        if best_example:
            # Adapt the best example to current question
            adapted_sql = self._adapt_example_to_question(best_example['sql'], question)
            return adapted_sql
        
        # Fallback: use knowledge base
        similar_queries = self.knowledge_base.search_queries(question.split()[:2])
        if similar_queries:
            return similar_queries[0]['sql']
        
        # Final fallback
        return "SELECT COUNT(*) FROM customers"
    
    def _find_best_example(self, question: str, examples: List[Dict]) -> Optional[Dict]:
        """Find the most relevant example for the question."""
        
        question_lower = question.lower()
        best_score = 0
        best_example = None
        
        for example in examples:
            score = self._calculate_example_relevance(question_lower, example)
            if score > best_score:
                best_score = score
                best_example = example
        
        return best_example if best_score > 0.3 else None
    
    def _calculate_example_relevance(self, question: str, example: Dict) -> float:
        """Calculate how relevant an example is to the question."""
        
        score = 0.0
        example_desc = example['question'].lower()
        
        # Keyword overlap
        question_words = set(question.split())
        example_words = set(example_desc.split())
        common_words = question_words.intersection(example_words)
        
        if common_words:
            score += len(common_words) / max(len(question_words), len(example_words))
        
        # Category relevance
        if example['category'] in ['count_with_filter', 'aggregation'] and any(word in question for word in ['count', 'how many', 'average']):
            score += 0.3
        
        if example['category'] == 'join' and any(word in question for word in ['customer', 'order', 'product']):
            score += 0.3
        
        return score
    
    def _adapt_example_to_question(self, example_sql: str, question: str) -> str:
        """Adapt an example SQL to fit the current question."""
        
        adapted = example_sql
        
        # Replace with relevant values from question
        relevant_values = self.value_retrieval.retrieve_relevant_values(question)
        
        for keyword, matches in relevant_values.items():
            if matches:
                value = matches[0]['value']
                table = matches[0]['table']
                column = matches[0]['column']
                
                # Replace placeholder values
                if 'Electronics' in adapted:
                    adapted = adapted.replace('Electronics', value)
                elif f'{table}.{column}' not in adapted and table in adapted:
                    # Add condition if table is already in query
                    if 'WHERE' in adapted:
                        adapted += f" AND {table}.{column} = '{value}'"
                    else:
                        adapted += f" WHERE {table}.{column} = '{value}'"
        
        return adapted
    
    def _calculate_synthetic_confidence(self, examples: List[Dict], sql: str) -> float:
        """Calculate confidence based on example quality."""
        confidence = 0.75  # Base confidence
        
        # Boost for knowledge base examples
        kb_examples = [ex for ex in examples if ex['relevance'] == 'knowledge_base_match']
        confidence += len(kb_examples) * 0.05
        
        # Boost for pattern matches
        pattern_examples = [ex for ex in examples if ex['relevance'] == 'pattern_match']
        confidence += len(pattern_examples) * 0.03
        
        return min(confidence, 1.0)


class LLMGenerator(BaseCandidateGenerator):
    """Generator using LLM (Gemini) for SQL generation."""
    
    def __init__(self, db_path: str, knowledge_base: QueryKnowledgeBase):
        super().__init__(db_path, knowledge_base)
        from llm_client import GeminiClient
        self.llm_client = GeminiClient()
    
    def generate_candidate(self, question: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SQL candidate using LLM."""
        
        if not self.llm_client.is_available():
            raise Exception("LLM client not available")
        
        sql = self.llm_client.generate_sql(question, schema)
        if not sql:
            raise Exception("Failed to generate SQL with LLM")
        
        return {
            'sql': sql,
            'confidence': 0.85,  # High confidence for LLM
            'reasoning': {
                'approach': 'llm',
                'method': 'Gemini API text-to-SQL generation',
                'features': ['natural_language_understanding', 'pattern_recognition']
            }
        }