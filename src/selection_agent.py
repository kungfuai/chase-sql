"""
Selection Agent module implementing pairwise comparison for candidate ranking.
Based on the CHASE-SQL paper's approach to selecting the best SQL query.
"""

import sqlite3
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np

class SelectionAgent:
    """
    Implements pairwise comparison-based selection of SQL candidates.
    Uses tournament-style scoring to identify the best query.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.comparison_cache = {}  # Cache pairwise comparisons
    
    def select_best_candidate(self, candidates: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
        """
        Select the best SQL candidate from a list using pairwise comparisons.
        
        Args:
            candidates: List of candidate queries with their metadata
            question: The original natural language question
            
        Returns:
            The best candidate with selection details
        """
        
        if len(candidates) == 0:
            raise ValueError("No candidates provided")
        
        if len(candidates) == 1:
            return {
                'selected_candidate': candidates[0],
                'selection_details': {
                    'total_candidates': 1,
                    'comparison_matrix': None,
                    'scores': [1.0],
                    'reasoning': 'Only one candidate available'
                }
            }
        
        # Build comparison matrix
        n = len(candidates)
        comparison_matrix = np.zeros((n, n))
        scores = [0.0] * n
        
        # Perform pairwise comparisons
        for i in range(n):
            for j in range(i + 1, n):
                winner_idx = self._compare_candidates(
                    candidates[i], 
                    candidates[j], 
                    question,
                    i, j
                )
                
                if winner_idx == i:
                    comparison_matrix[i][j] = 1
                    comparison_matrix[j][i] = 0
                    scores[i] += 1
                else:
                    comparison_matrix[i][j] = 0
                    comparison_matrix[j][i] = 1
                    scores[j] += 1
        
        # Find candidate with highest score
        best_idx = np.argmax(scores)
        
        return {
            'selected_candidate': candidates[best_idx],
            'selection_details': {
                'total_candidates': n,
                'comparison_matrix': comparison_matrix.tolist(),
                'scores': scores,
                'winner_index': best_idx,
                'reasoning': self._generate_selection_reasoning(candidates, scores, best_idx)
            }
        }
    
    def _compare_candidates(self, candidate1: Dict, candidate2: Dict, 
                          question: str, idx1: int, idx2: int) -> int:
        """
        Compare two candidates and return the index of the better one.
        
        Returns:
            Index of the winning candidate (idx1 or idx2)
        """
        
        # Check cache first
        cache_key = (candidate1['sql'], candidate2['sql'])
        if cache_key in self.comparison_cache:
            return self.comparison_cache[cache_key]
        
        # Execute both queries
        result1 = self._execute_candidate(candidate1['sql'])
        result2 = self._execute_candidate(candidate2['sql'])
        
        # If execution results are the same, use other criteria
        if result1['success'] == result2['success'] and result1.get('results') == result2.get('results'):
            winner_idx = self._compare_by_quality(candidate1, candidate2, question, idx1, idx2)
        else:
            winner_idx = self._compare_by_execution(result1, result2, candidate1, candidate2, 
                                                   question, idx1, idx2)
        
        # Cache the result
        self.comparison_cache[cache_key] = winner_idx
        
        return winner_idx
    
    def _execute_candidate(self, sql: str) -> Dict[str, Any]:
        """Execute a SQL query and return results."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(sql)
            results = cursor.fetchall()
            conn.close()
            
            return {
                'success': True,
                'results': results,
                'row_count': len(results),
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'results': None,
                'row_count': 0,
                'error': str(e)
            }
    
    def _compare_by_execution(self, result1: Dict, result2: Dict, 
                            candidate1: Dict, candidate2: Dict,
                            question: str, idx1: int, idx2: int) -> int:
        """Compare candidates based on execution results."""
        
        # Prefer successful queries
        if result1['success'] and not result2['success']:
            return idx1
        elif result2['success'] and not result1['success']:
            return idx2
        
        # Both successful - compare result quality
        if result1['success'] and result2['success']:
            # Prefer non-empty results
            if result1['row_count'] > 0 and result2['row_count'] == 0:
                return idx1
            elif result2['row_count'] > 0 and result1['row_count'] == 0:
                return idx2
            
            # Both have results - use quality metrics
            return self._compare_by_quality(candidate1, candidate2, question, idx1, idx2)
        
        # Both failed - compare error types
        return self._compare_failed_queries(result1, result2, idx1, idx2)
    
    def _compare_by_quality(self, candidate1: Dict, candidate2: Dict, 
                          question: str, idx1: int, idx2: int) -> int:
        """Compare candidates based on query quality metrics."""
        
        score1 = self._calculate_quality_score(candidate1, question)
        score2 = self._calculate_quality_score(candidate2, question)
        
        return idx1 if score1 >= score2 else idx2
    
    def _calculate_quality_score(self, candidate: Dict, question: str) -> float:
        """Calculate quality score for a candidate."""
        
        score = 0.0
        sql = candidate['sql'].upper()
        question_lower = question.lower()
        
        # Confidence from generator
        if 'confidence' in candidate:
            score += candidate['confidence'] * 10
        
        # Query complexity alignment with question
        if 'join' in question_lower and 'JOIN' in sql:
            score += 2
        elif 'join' not in question_lower and 'JOIN' not in sql:
            score += 1
        
        # Aggregation alignment
        agg_keywords = ['count', 'average', 'sum', 'total', 'maximum', 'minimum']
        agg_functions = ['COUNT', 'AVG', 'SUM', 'MAX', 'MIN']
        
        question_needs_agg = any(kw in question_lower for kw in agg_keywords)
        query_has_agg = any(func in sql for func in agg_functions)
        
        if question_needs_agg == query_has_agg:
            score += 2
        
        # Filtering alignment
        if 'where' in question_lower or 'filter' in question_lower:
            if 'WHERE' in sql:
                score += 1.5
        
        # Ordering alignment
        if any(word in question_lower for word in ['top', 'highest', 'lowest', 'best', 'worst']):
            if 'ORDER BY' in sql:
                score += 1.5
            if 'LIMIT' in sql:
                score += 1
        
        # Generator type preferences (based on paper insights)
        reasoning = candidate.get('reasoning', {})
        approach = reasoning.get('approach', '')
        
        if approach == 'divide_and_conquer' and any(word in question_lower for word in ['complex', 'multiple', 'and']):
            score += 1
        elif approach == 'query_plan' and 'join' in question_lower:
            score += 1
        elif approach == 'synthetic_examples':
            score += 0.5  # Slight preference for synthetic examples
        
        # Penalize overly simple queries for complex questions
        question_words = len(question.split())
        if question_words > 10 and sql.count('FROM') == 1 and 'JOIN' not in sql:
            score -= 1
        
        return score
    
    def _compare_failed_queries(self, result1: Dict, result2: Dict, idx1: int, idx2: int) -> int:
        """Compare two failed queries based on error types."""
        
        error1 = result1.get('error', '').lower()
        error2 = result2.get('error', '').lower()
        
        # Prefer syntax errors over semantic errors (easier to fix)
        syntax_error_keywords = ['syntax error', 'parse error']
        semantic_error_keywords = ['no such table', 'no such column', 'ambiguous']
        
        is_syntax1 = any(kw in error1 for kw in syntax_error_keywords)
        is_syntax2 = any(kw in error2 for kw in syntax_error_keywords)
        is_semantic1 = any(kw in error1 for kw in semantic_error_keywords)
        is_semantic2 = any(kw in error2 for kw in semantic_error_keywords)
        
        if is_syntax1 and is_semantic2:
            return idx1
        elif is_syntax2 and is_semantic1:
            return idx2
        
        # Default to first candidate if no clear winner
        return idx1
    
    def _generate_selection_reasoning(self, candidates: List[Dict], scores: List[float], 
                                    winner_idx: int) -> str:
        """Generate explanation for why a candidate was selected."""
        
        winner = candidates[winner_idx]
        winner_score = scores[winner_idx]
        total_comparisons = len(candidates) - 1
        
        reasoning_parts = [
            f"Selected candidate {winner_idx + 1} out of {len(candidates)} candidates.",
            f"Won {int(winner_score)} out of {total_comparisons} pairwise comparisons."
        ]
        
        # Add approach-specific reasoning
        approach = winner.get('reasoning', {}).get('approach', 'unknown')
        if approach == 'divide_and_conquer':
            reasoning_parts.append("Used divide-and-conquer approach which broke down the complex query.")
        elif approach == 'query_plan':
            reasoning_parts.append("Used query plan approach which mirrors database execution steps.")
        elif approach == 'synthetic_examples':
            reasoning_parts.append("Used synthetic examples to guide query generation.")
        
        # Add confidence info
        if 'confidence' in winner:
            reasoning_parts.append(f"Generator confidence: {winner['confidence']:.2f}")
        
        return " ".join(reasoning_parts)
    
    def get_selection_matrix_visualization(self, comparison_matrix: List[List[float]], 
                                         candidates: List[Dict]) -> str:
        """Create a text visualization of the comparison matrix."""
        
        n = len(candidates)
        lines = ["Pairwise Comparison Matrix:"]
        lines.append("(1 = row candidate won, 0 = column candidate won)")
        lines.append("")
        
        # Header
        header = "   "
        for i in range(n):
            header += f" C{i+1} "
        lines.append(header)
        
        # Matrix rows
        for i in range(n):
            row = f"C{i+1} "
            for j in range(n):
                if i == j:
                    row += " -  "
                else:
                    row += f" {int(comparison_matrix[i][j])}  "
            lines.append(row)
        
        return "\n".join(lines)