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
    LLM-powered Divide and Conquer Chain-of-Thought generator.
    Uses Gemini API to break down complex queries into manageable sub-problems.
    """
    
    def __init__(self, db_path: str, knowledge_base: QueryKnowledgeBase):
        super().__init__(db_path, knowledge_base)
        from llm_client import GeminiClient
        self.llm_client = GeminiClient()
    
    def generate_candidate(self, question: str, schema: str) -> Dict[str, Any]:
        """Generate SQL using LLM-powered divide and conquer approach."""
        
        if not self.llm_client.is_available():
            # Fallback to rule-based approach if LLM unavailable
            return self._fallback_generate(question, schema)
        
        # Step 1: LLM-powered question analysis and decomposition
        decomposition_result = self._llm_decompose_question(question, schema)
        
        # Step 2: Generate sub-solutions using LLM
        sub_solutions = self._llm_solve_subproblems(decomposition_result, schema)
        
        # Step 3: Combine solutions using LLM
        final_sql = self._llm_combine_solutions(question, sub_solutions, schema)
        
        return {
            'sql': final_sql,
            'reasoning': {
                'approach': 'llm_divide_and_conquer',
                'decomposition': decomposition_result,
                'sub_solutions': sub_solutions,
                'llm_reasoning': 'Used Chain-of-Thought decomposition with Gemini API'
            },
            'confidence': 0.88  # Higher confidence for LLM approach
        }
    
    def _llm_decompose_question(self, question: str, schema: str) -> Dict[str, Any]:
        """Use LLM to decompose the question into sub-problems."""
        
        # Get some context from value retrieval
        relevant_values = self.value_retrieval.retrieve_relevant_values(question)
        context = self._format_value_context(relevant_values)
        
        prompt = f"""You are an expert at breaking down complex SQL questions into simpler sub-problems.

Database Schema:
{schema}

Context (relevant values found in database):
{context}

Question: {question}

Please analyze this question and break it down using divide-and-conquer approach. Think step by step:

1. ANALYSIS: What is the main intent? What entities are involved? What constraints/filters are needed?
2. DECOMPOSITION: Break this into 2-4 logical sub-problems that can be solved independently
3. DEPENDENCIES: What order should these sub-problems be solved in?

Format your response as:
ANALYSIS:
- Main Intent: [count/retrieve/aggregate/etc]
- Entities: [tables involved]  
- Constraints: [filters needed]
- Aggregations: [any aggregation functions]

DECOMPOSITION:
1. [Sub-problem 1 description]
2. [Sub-problem 2 description]
3. [Sub-problem 3 description if needed]

DEPENDENCIES:
[Explain the logical order and dependencies between sub-problems]"""

        try:
            response = self.llm_client._client.generate_content(prompt)
            return self._parse_decomposition_response(response.text, question)
        except Exception:
            return self._fallback_analyze_question(question)
    
    def _llm_solve_subproblems(self, decomposition: Dict[str, Any], schema: str) -> List[Dict[str, Any]]:
        """Use LLM to solve each sub-problem."""
        
        sub_solutions = []
        
        for i, sub_problem in enumerate(decomposition.get('sub_problems', [])):
            prompt = f"""You are solving sub-problem {i+1} of a divide-and-conquer SQL generation task.

Database Schema:
{schema}

Overall Question Context: {decomposition.get('original_question', '')}
Analysis: {decomposition.get('analysis', {})}

Sub-problem to solve: {sub_problem}

Previous sub-solutions for context:
{self._format_previous_solutions(sub_solutions)}

Generate the SQL component or logic needed for this specific sub-problem. 
Focus ONLY on this sub-problem, not the complete query.

Respond with:
APPROACH: [How you're solving this sub-problem]
SQL_COMPONENT: [The SQL piece for this sub-problem]
EXPLANATION: [Brief explanation of this component]"""

            try:
                response = self.llm_client._client.generate_content(prompt)
                solution = self._parse_subproblem_response(response.text, sub_problem)
                sub_solutions.append(solution)
            except Exception:
                # Fallback solution
                sub_solutions.append({
                    'sub_problem': sub_problem,
                    'approach': 'fallback',
                    'sql_component': 'SELECT * FROM table',
                    'explanation': 'Fallback solution'
                })
        
        return sub_solutions
    
    def _llm_combine_solutions(self, question: str, sub_solutions: List[Dict], schema: str) -> str:
        """Use LLM to combine sub-solutions into final SQL."""
        
        solutions_text = "\n".join([
            f"Sub-problem: {sol['sub_problem']}\nSQL Component: {sol['sql_component']}\nApproach: {sol['approach']}"
            for sol in sub_solutions
        ])
        
        prompt = f"""You are combining sub-solutions into a complete SQL query.

Database Schema:
{schema}

Original Question: {question}

Sub-solutions to combine:
{solutions_text}

Now combine these sub-solutions into a single, complete, executable SQL query.
Make sure to:
1. Use proper JOIN syntax if multiple tables are involved
2. Apply all necessary WHERE conditions  
3. Use correct aggregation functions
4. Ensure proper ORDER BY and LIMIT clauses where needed
5. Make the query syntactically correct

Return ONLY the final SQL query, no explanation."""

        try:
            response = self.llm_client._client.generate_content(prompt)
            sql = response.text.strip()
            
            # Clean up the response
            if sql.startswith('```sql'):
                sql = sql[6:]
            if sql.endswith('```'):
                sql = sql[:-3]
            return sql.strip()
            
        except Exception:
            # Fallback: use first solution or basic query
            if sub_solutions:
                return sub_solutions[0].get('sql_component', 'SELECT COUNT(*) FROM customers')
            return 'SELECT COUNT(*) FROM customers'
    
    def _format_value_context(self, relevant_values: Dict) -> str:
        """Format relevant values for context."""
        if not relevant_values:
            return "No specific values found in database for this question."
        
        lines = []
        for keyword, matches in relevant_values.items():
            for match in matches[:2]:  # Top 2 matches per keyword
                lines.append(f"- '{keyword}' maps to {match['table']}.{match['column']} = '{match['value']}'")
        
        return "\n".join(lines) if lines else "No specific value mappings found."
    
    def _parse_decomposition_response(self, response: str, question: str) -> Dict[str, Any]:
        """Parse LLM decomposition response."""
        
        result = {
            'original_question': question,
            'analysis': {},
            'sub_problems': [],
            'dependencies': ''
        }
        
        # Simple parsing - in production would be more robust
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('ANALYSIS:'):
                current_section = 'analysis'
            elif line.startswith('DECOMPOSITION:'):
                current_section = 'decomposition'
            elif line.startswith('DEPENDENCIES:'):
                current_section = 'dependencies'
            elif line and current_section:
                if current_section == 'decomposition' and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.') or line.startswith('4.')):
                    result['sub_problems'].append(line[2:].strip())
                elif current_section == 'dependencies':
                    result['dependencies'] += line + ' '
                elif current_section == 'analysis' and ':' in line:
                    key, value = line.split(':', 1)
                    result['analysis'][key.strip('- ')] = value.strip()
        
        # Fallback if parsing failed
        if not result['sub_problems']:
            result['sub_problems'] = [
                "Identify the main entities and tables needed",
                "Determine filtering conditions",
                "Apply aggregation or selection logic"
            ]
        
        return result
    
    def _parse_subproblem_response(self, response: str, sub_problem: str) -> Dict[str, Any]:
        """Parse LLM sub-problem response."""
        
        result = {
            'sub_problem': sub_problem,
            'approach': '',
            'sql_component': '',
            'explanation': ''
        }
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('APPROACH:'):
                current_section = 'approach'
                result['approach'] = line[9:].strip()
            elif line.startswith('SQL_COMPONENT:'):
                current_section = 'sql_component'
                result['sql_component'] = line[14:].strip()
            elif line.startswith('EXPLANATION:'):
                current_section = 'explanation'
                result['explanation'] = line[12:].strip()
            elif line and current_section:
                result[current_section] += ' ' + line
        
        # Clean up SQL component
        sql = result['sql_component']
        if sql.startswith('```sql'):
            sql = sql[6:]
        if sql.endswith('```'):
            sql = sql[:-3]
        result['sql_component'] = sql.strip()
        
        return result
    
    def _format_previous_solutions(self, solutions: List[Dict]) -> str:
        """Format previous solutions for context."""
        if not solutions:
            return "No previous solutions yet."
        
        formatted = []
        for i, sol in enumerate(solutions):
            formatted.append(f"{i+1}. {sol['sub_problem']}: {sol['sql_component']}")
        
        return "\n".join(formatted)
    
    def _fallback_generate(self, question: str, schema: str) -> Dict[str, Any]:
        """Fallback to rule-based approach when LLM unavailable."""
        analysis = self._fallback_analyze_question(question)
        sql = self._construct_basic_query(question, schema)
        
        return {
            'sql': sql,
            'reasoning': {
                'approach': 'divide_and_conquer_fallback',
                'analysis': analysis,
                'note': 'Used rule-based fallback due to LLM unavailability'
            },
            'confidence': 0.65
        }
    
    def _fallback_analyze_question(self, question: str) -> Dict[str, Any]:
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
    LLM-powered Query Plan Chain-of-Thought generator.
    Uses Gemini API to mirror database execution steps and generate SQL queries.
    """
    
    def __init__(self, db_path: str, knowledge_base: QueryKnowledgeBase):
        super().__init__(db_path, knowledge_base)
        from llm_client import GeminiClient
        self.llm_client = GeminiClient()
    
    def generate_candidate(self, question: str, schema: str) -> Dict[str, Any]:
        """Generate SQL using LLM-powered query execution plan approach."""
        
        if not self.llm_client.is_available():
            # Fallback to rule-based approach
            return self._fallback_plan_generate(question, schema)
        
        # Step 1: LLM creates detailed execution plan
        execution_plan = self._llm_create_execution_plan(question, schema)
        
        # Step 2: LLM converts plan to optimized SQL
        sql_query = self._llm_plan_to_sql(execution_plan, question, schema)
        
        return {
            'sql': sql_query,
            'reasoning': {
                'approach': 'llm_query_plan',
                'execution_plan': execution_plan,
                'llm_reasoning': 'Used database execution plan simulation with Gemini API'
            },
            'confidence': 0.87  # High confidence for LLM approach
        }
    
    def _llm_create_execution_plan(self, question: str, schema: str) -> Dict[str, Any]:
        """Use LLM to create a detailed query execution plan."""
        
        # Get relevant values for context
        relevant_values = self.value_retrieval.retrieve_relevant_values(question)
        context = self._format_value_context(relevant_values)
        
        prompt = f"""You are a database query optimizer creating an execution plan.

Database Schema:
{schema}

Context (relevant values found in database):
{context}

Question: {question}

Create a detailed query execution plan as if you were a database engine. Think step by step about how a database would execute this query:

1. TABLE SCAN: Which tables need to be scanned? In what order?
2. JOIN OPERATIONS: What joins are needed? What join algorithms would be used?
3. FILTERING: What WHERE conditions should be applied? When in the execution?
4. AGGREGATION: What aggregation operations are needed? GROUP BY requirements?
5. SORTING: Is sorting needed? What columns and order?
6. LIMITING: Any LIMIT or TOP clauses needed?
7. OPTIMIZATION: What indexes might be used? Any query optimizations?

Format your response as:
TABLES_TO_SCAN:
- [table1]: [reason for scanning]
- [table2]: [reason for scanning]

JOIN_OPERATIONS:
- [join description with tables and conditions]

FILTERING_STEPS:
- [filter condition 1]: [when to apply]
- [filter condition 2]: [when to apply]

AGGREGATION_STEPS:
- [aggregation operation]: [grouping requirements]

SORTING_REQUIREMENTS:
- [sort column]: [ASC/DESC] [reason]

LIMIT_CLAUSES:
- [any limiting requirements]

EXECUTION_ORDER:
1. [Step 1]
2. [Step 2]
3. [Step 3]
..."""

        try:
            response = self.llm_client._client.generate_content(prompt)
            return self._parse_execution_plan_response(response.text, question)
        except Exception:
            return self._fallback_create_execution_plan(question, schema)
    
    def _llm_plan_to_sql(self, execution_plan: Dict[str, Any], question: str, schema: str) -> str:
        """Use LLM to convert execution plan to optimized SQL."""
        
        plan_text = self._format_execution_plan(execution_plan)
        
        prompt = f"""You are converting a database execution plan into optimized SQL.

Database Schema:
{schema}

Original Question: {question}

Execution Plan:
{plan_text}

Convert this execution plan into a single, optimized SQL query. Follow the execution plan steps precisely:
1. Use the exact tables specified in the plan
2. Apply joins in the order and manner specified
3. Add WHERE conditions as planned
4. Include aggregation and GROUP BY as needed
5. Add ORDER BY and LIMIT clauses as specified
6. Optimize for performance (use appropriate indexes conceptually)

Important: Generate syntactically correct SQL that follows standard SQL syntax.

Return ONLY the SQL query, no explanation or formatting."""

        try:
            response = self.llm_client._client.generate_content(prompt)
            sql = response.text.strip()
            
            # Clean up response
            if sql.startswith('```sql'):
                sql = sql[6:]
            if sql.endswith('```'):
                sql = sql[:-3]
            return sql.strip()
            
        except Exception:
            return self._fallback_plan_to_sql(execution_plan, question)
    
    def _parse_execution_plan_response(self, response: str, question: str) -> Dict[str, Any]:
        """Parse LLM execution plan response."""
        
        plan = {
            'original_question': question,
            'tables_to_scan': [],
            'join_operations': [],
            'filtering_steps': [],
            'aggregation_steps': [],
            'sorting_requirements': [],
            'limit_clauses': [],
            'execution_order': []
        }
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('TABLES_TO_SCAN:'):
                current_section = 'tables_to_scan'
            elif line.startswith('JOIN_OPERATIONS:'):
                current_section = 'join_operations'
            elif line.startswith('FILTERING_STEPS:'):
                current_section = 'filtering_steps'
            elif line.startswith('AGGREGATION_STEPS:'):
                current_section = 'aggregation_steps'
            elif line.startswith('SORTING_REQUIREMENTS:'):
                current_section = 'sorting_requirements'
            elif line.startswith('LIMIT_CLAUSES:'):
                current_section = 'limit_clauses'
            elif line.startswith('EXECUTION_ORDER:'):
                current_section = 'execution_order'
            elif line and current_section and line.startswith('- '):
                plan[current_section].append(line[2:])
            elif line and current_section == 'execution_order' and (line[0].isdigit() and line[1] == '.'):
                plan[current_section].append(line[2:].strip())
        
        # Ensure we have some default values
        if not plan['tables_to_scan']:
            plan['tables_to_scan'] = ['customers: default table for queries']
        if not plan['execution_order']:
            plan['execution_order'] = ['Scan tables', 'Apply filters', 'Format results']
        
        return plan
    
    def _format_execution_plan(self, plan: Dict[str, Any]) -> str:
        """Format execution plan for LLM consumption."""
        
        sections = []
        
        if plan.get('tables_to_scan'):
            sections.append("Tables to scan:\n" + "\n".join(f"- {table}" for table in plan['tables_to_scan']))
        
        if plan.get('join_operations'):
            sections.append("Join operations:\n" + "\n".join(f"- {join}" for join in plan['join_operations']))
        
        if plan.get('filtering_steps'):
            sections.append("Filtering steps:\n" + "\n".join(f"- {filter_step}" for filter_step in plan['filtering_steps']))
        
        if plan.get('aggregation_steps'):
            sections.append("Aggregation steps:\n" + "\n".join(f"- {agg}" for agg in plan['aggregation_steps']))
        
        if plan.get('sorting_requirements'):
            sections.append("Sorting requirements:\n" + "\n".join(f"- {sort}" for sort in plan['sorting_requirements']))
        
        if plan.get('execution_order'):
            sections.append("Execution order:\n" + "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan['execution_order'])))
        
        return "\n\n".join(sections)
    
    def _format_value_context(self, relevant_values: Dict) -> str:
        """Format relevant values for context."""
        if not relevant_values:
            return "No specific values found in database for this question."
        
        lines = []
        for keyword, matches in relevant_values.items():
            for match in matches[:2]:  # Top 2 matches per keyword
                lines.append(f"- '{keyword}' maps to {match['table']}.{match['column']} = '{match['value']}'")
        
        return "\n".join(lines) if lines else "No specific value mappings found."
    
    def _fallback_plan_generate(self, question: str, schema: str) -> Dict[str, Any]:
        """Fallback to rule-based approach when LLM unavailable."""
        execution_plan = self._fallback_create_execution_plan(question, schema)
        sql_query = self._fallback_plan_to_sql(execution_plan, question)
        
        return {
            'sql': sql_query,
            'reasoning': {
                'approach': 'query_plan_fallback',
                'execution_plan': execution_plan,
                'note': 'Used rule-based fallback due to LLM unavailability'
            },
            'confidence': 0.68
        }
    
    def _fallback_create_execution_plan(self, question: str, schema: str) -> Dict[str, Any]:
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
    
    def _fallback_plan_to_sql(self, plan: Dict[str, Any], question: str) -> str:
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
    LLM-powered Online Synthetic Example Generation.
    Uses Gemini API to generate relevant few-shot examples on-the-fly to guide SQL generation.
    """
    
    def __init__(self, db_path: str, knowledge_base: QueryKnowledgeBase):
        super().__init__(db_path, knowledge_base)
        from llm_client import GeminiClient
        self.llm_client = GeminiClient()
    
    def generate_candidate(self, question: str, schema: str) -> Dict[str, Any]:
        """Generate SQL using LLM-powered synthetic examples approach."""
        
        if not self.llm_client.is_available():
            # Fallback to rule-based approach
            return self._fallback_synthetic_generate(question, schema)
        
        # Step 1: LLM generates high-quality synthetic examples
        synthetic_examples = self._llm_generate_synthetic_examples(question, schema)
        
        # Step 2: LLM uses examples to guide SQL generation with few-shot learning
        sql_query = self._llm_generate_from_examples(question, synthetic_examples, schema)
        
        return {
            'sql': sql_query,
            'reasoning': {
                'approach': 'llm_synthetic_examples',
                'generated_examples': synthetic_examples,
                'llm_reasoning': 'Used LLM-generated few-shot examples with Gemini API'
            },
            'confidence': 0.86  # High confidence for LLM approach
        }
    
    def _llm_generate_synthetic_examples(self, question: str, schema: str, num_examples: int = 3) -> List[Dict]:
        """Use LLM to generate high-quality synthetic examples."""
        
        # Get some context from knowledge base and value retrieval
        question_keywords = question.lower().split()
        similar_queries = self.knowledge_base.search_queries(question_keywords[:3])
        relevant_values = self.value_retrieval.retrieve_relevant_values(question)
        
        context_examples = []
        if similar_queries:
            context_examples = similar_queries[:2]  # Top 2 similar queries
        
        prompt = f"""You are an expert at generating relevant few-shot examples for text-to-SQL tasks.

Database Schema:
{schema}

Target Question: {question}

Context (existing similar queries in knowledge base):
{self._format_context_examples(context_examples)}

Relevant values found in database:
{self._format_value_context_synthetic(relevant_values)}

Generate {num_examples} high-quality synthetic examples that would help a model learn to convert the target question into SQL. Each example should:

1. Be similar in structure/complexity to the target question
2. Use the same database schema
3. Demonstrate the same type of SQL operations needed
4. Have progressive difficulty leading up to the target question
5. Use realistic data values that might exist in the database

Format your response as:
EXAMPLE_1:
Question: [Natural language question]
SQL: [Corresponding SQL query]
Explanation: [Why this example is relevant]

EXAMPLE_2:
Question: [Natural language question]  
SQL: [Corresponding SQL query]
Explanation: [Why this example is relevant]

EXAMPLE_3:
Question: [Natural language question]
SQL: [Corresponding SQL query] 
Explanation: [Why this example is relevant]"""

        try:
            response = self.llm_client._client.generate_content(prompt)
            return self._parse_synthetic_examples_response(response.text)
        except Exception:
            return self._fallback_generate_synthetic_examples(question, schema, num_examples)
    
    def _llm_generate_from_examples(self, question: str, examples: List[Dict], schema: str) -> str:
        """Use LLM with few-shot examples to generate SQL."""
        
        examples_text = self._format_examples_for_prompt(examples)
        
        prompt = f"""You are converting natural language questions to SQL using few-shot learning.

Database Schema:
{schema}

Here are some relevant examples to learn from:

{examples_text}

Now convert this question to SQL following the patterns shown in the examples:

Question: {question}

Generate a syntactically correct SQL query that follows the same patterns and style as the examples above.

Return ONLY the SQL query, no explanation."""

        try:
            response = self.llm_client._client.generate_content(prompt)
            sql = response.text.strip()
            
            # Clean up response
            if sql.startswith('```sql'):
                sql = sql[6:]
            if sql.endswith('```'):
                sql = sql[:-3]
            return sql.strip()
            
        except Exception:
            return self._fallback_generate_from_examples(question, examples)
    
    def _parse_synthetic_examples_response(self, response: str) -> List[Dict]:
        """Parse LLM synthetic examples response."""
        
        examples = []
        lines = response.split('\n')
        current_example = {}
        current_field = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('EXAMPLE_'):
                if current_example:
                    examples.append(current_example)
                current_example = {'relevance': 'llm_generated'}
            elif line.startswith('Question:'):
                current_field = 'question'
                current_example['question'] = line[9:].strip()
            elif line.startswith('SQL:'):
                current_field = 'sql'
                sql = line[4:].strip()
                # Clean up SQL
                if sql.startswith('```sql'):
                    sql = sql[6:]
                if sql.endswith('```'):
                    sql = sql[:-3]
                current_example['sql'] = sql.strip()
            elif line.startswith('Explanation:'):
                current_field = 'explanation'
                current_example['explanation'] = line[12:].strip()
            elif line and current_field and current_example:
                # Continue previous field
                current_example[current_field] += ' ' + line
        
        # Add the last example
        if current_example:
            examples.append(current_example)
        
        # Fallback if parsing failed
        if not examples:
            examples = [{
                'question': 'How many customers are there?',
                'sql': 'SELECT COUNT(*) FROM customers',
                'explanation': 'Basic count query',
                'relevance': 'fallback'
            }]
        
        return examples
    
    def _format_context_examples(self, similar_queries: List[Dict]) -> str:
        """Format context examples from knowledge base."""
        if not similar_queries:
            return "No similar queries found in knowledge base."
        
        formatted = []
        for i, query in enumerate(similar_queries):
            formatted.append(f"{i+1}. Question: {query.get('description', 'N/A')}")
            formatted.append(f"   SQL: {query.get('sql', 'N/A')}")
            formatted.append(f"   Category: {query.get('category', 'N/A')}")
        
        return "\n".join(formatted)
    
    def _format_value_context_synthetic(self, relevant_values: Dict) -> str:
        """Format relevant values for synthetic generation."""
        if not relevant_values:
            return "No specific values found in database for this question."
        
        lines = []
        for keyword, matches in relevant_values.items():
            for match in matches[:2]:  # Top 2 matches per keyword
                lines.append(f"- '{keyword}' maps to {match['table']}.{match['column']} = '{match['value']}'")
        
        return "\n".join(lines) if lines else "No specific value mappings found."
    
    def _format_examples_for_prompt(self, examples: List[Dict]) -> str:
        """Format examples for few-shot learning prompt."""
        
        formatted = []
        for i, example in enumerate(examples):
            formatted.append(f"Example {i+1}:")
            formatted.append(f"Question: {example.get('question', '')}")
            formatted.append(f"SQL: {example.get('sql', '')}")
            if example.get('explanation'):
                formatted.append(f"Note: {example['explanation']}")
            formatted.append("")  # Empty line between examples
        
        return "\n".join(formatted)
    
    def _fallback_synthetic_generate(self, question: str, schema: str) -> Dict[str, Any]:
        """Fallback to rule-based approach when LLM unavailable."""
        synthetic_examples = self._fallback_generate_synthetic_examples(question, schema)
        sql_query = self._fallback_generate_from_examples(question, synthetic_examples)
        
        return {
            'sql': sql_query,
            'reasoning': {
                'approach': 'synthetic_examples_fallback',
                'generated_examples': synthetic_examples,
                'note': 'Used rule-based fallback due to LLM unavailability'
            },
            'confidence': 0.66
        }
    
    def _fallback_generate_synthetic_examples(self, question: str, schema: str, num_examples: int = 3) -> List[Dict]:
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
    
    def _fallback_generate_from_examples(self, question: str, examples: List[Dict]) -> str:
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