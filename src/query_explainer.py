"""
Query Explainer module for generating human-readable explanations of SQL queries.
Provides both inline SQL comments and plain English explanations.
"""

import re
from typing import Dict, Any, List, Tuple
from llm_client import GeminiClient

class QueryExplainer:
    """
    Generates explanations for SQL queries to help users understand and validate them.
    """
    
    def __init__(self):
        self.llm_client = GeminiClient()
    
    def explain_query(self, sql: str, question: str) -> Dict[str, Any]:
        """
        Generate both annotated SQL and plain English explanation for a query.
        
        Args:
            sql: The SQL query to explain
            question: The original natural language question
            
        Returns:
            Dict containing annotated SQL and plain English explanation
        """
        
        # Generate annotated SQL with inline comments
        annotated_sql = self._annotate_sql(sql, question)
        
        # Generate plain English explanation with numbered steps
        plain_explanation = self._generate_plain_explanation(sql, question)
        
        return {
            'original_sql': sql,
            'annotated_sql': annotated_sql,
            'plain_explanation': plain_explanation,
            'question': question
        }
    
    def _annotate_sql(self, sql: str, question: str) -> str:
        """
        Add inline comments to SQL query explaining each part.
        """
        
        # Check if LLM is available
        if not self.llm_client.is_available():
            # Fallback to simple rule-based annotation
            return self._fallback_annotate_sql(sql, question)
        
        prompt = f"""Given this SQL query that answers the question "{question}", 
add helpful inline comments to explain what each part does.

Original SQL:
{sql}

Rules:
1. Add comments using SQL comment syntax: -- for single line comments
2. Place comments at the end of important lines
3. Keep comments concise and clear
4. Focus on explaining the business logic, not SQL syntax
5. Preserve the exact SQL structure and formatting

Return ONLY the annotated SQL with comments, nothing else."""

        response = self.llm_client.generate(prompt)
        
        if not response:
            # Fallback if generation fails
            return self._fallback_annotate_sql(sql, question)
        
        # Clean up the response if needed
        annotated = response.strip()
        
        # Remove any markdown code blocks if present
        if annotated.startswith('```'):
            lines = annotated.split('\n')
            # Remove first and last lines if they're code block markers
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines[-1] == '```':
                lines = lines[:-1]
            annotated = '\n'.join(lines)
        
        return annotated
    
    def _generate_plain_explanation(self, sql: str, question: str) -> str:
        """
        Generate a plain English explanation with numbered steps.
        """
        
        # Check if LLM is available
        if not self.llm_client.is_available():
            # Fallback to simple rule-based explanation
            return self._fallback_plain_explanation(sql, question)
        
        prompt = f"""Explain this SQL query in plain English with numbered steps.
The query answers the question: "{question}"

SQL:
{sql}

Provide a clear, concise explanation that:
1. Uses numbered steps (1), (2), (3), etc.
2. Explains the logical flow of the query
3. Focuses on what the query does, not how SQL works
4. Is understandable by non-technical users

Format: Start with "This query finds/calculates/retrieves..." followed by numbered steps.
Keep it to one paragraph with inline numbering.

Return ONLY the explanation, nothing else."""

        response = self.llm_client.generate(prompt)
        
        if not response:
            # Fallback if generation fails
            return self._fallback_plain_explanation(sql, question)
        
        return response.strip()
    
    def get_execution_plan_explanation(self, sql: str) -> str:
        """
        Generate a simple explanation of how the database will execute this query.
        Useful for understanding performance implications.
        """
        
        explanation_parts = []
        
        # Analyze query structure
        sql_upper = sql.upper()
        
        # Check for main operations
        if 'JOIN' in sql_upper:
            join_count = sql_upper.count('JOIN')
            explanation_parts.append(f"performs {join_count} table join(s)")
        
        if 'GROUP BY' in sql_upper:
            explanation_parts.append("groups results for aggregation")
        
        if 'ORDER BY' in sql_upper:
            explanation_parts.append("sorts the results")
            
        if 'LIMIT' in sql_upper:
            limit_match = re.search(r'LIMIT\s+(\d+)', sql, re.IGNORECASE)
            if limit_match:
                limit_num = limit_match.group(1)
                explanation_parts.append(f"returns only top {limit_num} result(s)")
        
        if 'WHERE' in sql_upper:
            explanation_parts.append("filters rows based on conditions")
        
        # Combine parts
        if explanation_parts:
            return "This query " + ", ".join(explanation_parts) + "."
        else:
            return "This is a simple query with minimal processing."
    
    def validate_against_question(self, sql: str, question: str) -> Dict[str, Any]:
        """
        Check if the SQL query appears to correctly answer the question.
        """
        
        prompt = f"""Analyze if this SQL query correctly answers the given question.

Question: "{question}"

SQL Query:
{sql}

Evaluate:
1. Does the query address all parts of the question?
2. Are there any missing conditions or filters?
3. Is the logic correct for answering the question?

Respond with a JSON object containing:
- "is_valid": true/false
- "confidence": 0.0 to 1.0
- "issues": list of any problems found (empty list if none)
- "suggestions": list of improvements (empty list if none)

Return ONLY the JSON object."""

        response = self.llm_client.generate(prompt)
        
        # Parse JSON response
        import json
        try:
            validation = json.loads(response)
            return validation
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "is_valid": True,
                "confidence": 0.5,
                "issues": ["Could not parse validation response"],
                "suggestions": []
            }
    
    def _fallback_annotate_sql(self, sql: str, question: str) -> str:
        """
        Fallback method for SQL annotation when LLM is not available.
        Uses simple rule-based approach.
        """
        lines = sql.split('\n')
        annotated_lines = []
        
        for line in lines:
            line_upper = line.upper().strip()
            
            # Add comments based on SQL keywords
            if line_upper.startswith('SELECT'):
                if 'COUNT' in line_upper:
                    annotated_lines.append(line + '  -- Count records')
                elif 'SUM' in line_upper:
                    annotated_lines.append(line + '  -- Calculate total')
                elif 'AVG' in line_upper:
                    annotated_lines.append(line + '  -- Calculate average')
                elif 'MAX' in line_upper or 'MIN' in line_upper:
                    annotated_lines.append(line + '  -- Find extreme value')
                else:
                    annotated_lines.append(line + '  -- Select data')
            
            elif line_upper.startswith('FROM'):
                annotated_lines.append(line + '  -- Main table')
            
            elif 'JOIN' in line_upper:
                annotated_lines.append(line + '  -- Combine with related data')
            
            elif line_upper.startswith('WHERE'):
                annotated_lines.append(line + '  -- Filter conditions')
            
            elif line_upper.startswith('GROUP BY'):
                annotated_lines.append(line + '  -- Group for aggregation')
            
            elif line_upper.startswith('ORDER BY'):
                if 'DESC' in line_upper:
                    annotated_lines.append(line + '  -- Sort highest first')
                else:
                    annotated_lines.append(line + '  -- Sort results')
            
            elif line_upper.startswith('LIMIT'):
                annotated_lines.append(line + '  -- Limit results')
            
            else:
                annotated_lines.append(line)
        
        return '\n'.join(annotated_lines)
    
    def _fallback_plain_explanation(self, sql: str, question: str) -> str:
        """
        Fallback method for plain English explanation when LLM is not available.
        Uses simple pattern matching.
        """
        sql_upper = sql.upper()
        steps = []
        
        # Analyze query structure
        if 'SELECT' in sql_upper:
            if 'COUNT' in sql_upper:
                steps.append("counts records")
            elif 'SUM' in sql_upper:
                steps.append("calculates totals")
            elif 'AVG' in sql_upper:
                steps.append("calculates averages")
            else:
                steps.append("retrieves data")
        
        if 'JOIN' in sql_upper:
            join_count = sql_upper.count('JOIN')
            if join_count == 1:
                steps.append("combines data from related tables")
            else:
                steps.append(f"combines data from {join_count + 1} tables")
        
        if 'WHERE' in sql_upper:
            steps.append("filters based on conditions")
        
        if 'GROUP BY' in sql_upper:
            steps.append("groups results for aggregation")
        
        if 'ORDER BY' in sql_upper:
            if 'DESC' in sql_upper:
                steps.append("sorts by highest values first")
            else:
                steps.append("sorts the results")
        
        if 'LIMIT' in sql_upper:
            limit_match = re.search(r'LIMIT\s+(\d+)', sql, re.IGNORECASE)
            if limit_match:
                limit_num = limit_match.group(1)
                if limit_num == '1':
                    steps.append("returns only the top result")
                else:
                    steps.append(f"returns the top {limit_num} results")
        
        # Combine into explanation
        if steps:
            explanation = f"This query {steps[0]}"
            if len(steps) > 1:
                for i, step in enumerate(steps[1:], 1):
                    explanation += f", ({i + 1}) {step}"
            explanation += "."
            return explanation
        else:
            return "This query retrieves data based on the specified criteria."