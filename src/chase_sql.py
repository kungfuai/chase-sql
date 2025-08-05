"""
Main CHASE-SQL system integrating all components.
Implements the complete text-to-SQL pipeline from the paper.
"""

from typing import List, Dict, Any, Optional
import time
from database import ECommerceDB
from knowledge_base import QueryKnowledgeBase
from value_retrieval import ValueRetrieval
from generators import DivideConquerGenerator, QueryPlanGenerator, OnlineSyntheticGenerator, LLMGenerator
from config import Config
from query_fixer import QueryFixer
from selection_agent import SelectionAgent

class ChaseSQL:
    """
    Main CHASE-SQL system implementing multi-path reasoning and 
    preference-optimized candidate selection.
    """
    
    def __init__(self, db_path: str = "ecommerce.db"):
        # Initialize database
        self.db = ECommerceDB(db_path)
        self.db_path = db_path
        
        # Initialize components
        self.knowledge_base = QueryKnowledgeBase()
        self.value_retrieval = ValueRetrieval(db_path)
        
        # Initialize generators
        self.generators = {
            'divide_conquer': DivideConquerGenerator(db_path, self.knowledge_base),
            'query_plan': QueryPlanGenerator(db_path, self.knowledge_base),
            'synthetic': OnlineSyntheticGenerator(db_path, self.knowledge_base)
        }
        
        # Add LLM generator if available
        if Config.is_llm_enabled():
            try:
                self.generators['llm'] = LLMGenerator(db_path, self.knowledge_base)
            except Exception:
                pass  # LLM not available, continue with rule-based generators
        
        # Initialize fixer and selector
        self.query_fixer = QueryFixer(db_path)
        self.selection_agent = SelectionAgent(db_path)
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'generator_successes': {name: 0 for name in self.generators},
            'average_candidates': 0,
            'average_time': 0
        }
    
    def process_question(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Process a natural language question and return SQL query with full details.
        
        Args:
            question: Natural language question
            verbose: Whether to print intermediate steps
            
        Returns:
            Dictionary containing final SQL, execution results, and process details
        """
        
        start_time = time.time()
        self.stats['total_queries'] += 1
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"CHASE-SQL Processing: {question}")
            print(f"{'='*60}\n")
        
        # Step 1: Value Retrieval
        if verbose:
            print("Step 1: Value Retrieval")
            print("-" * 30)
        
        relevant_values = self.value_retrieval.retrieve_relevant_values(question)
        
        if verbose and relevant_values:
            print("Retrieved values:")
            for keyword, matches in list(relevant_values.items())[:3]:
                if matches:
                    match = matches[0]
                    print(f"  '{keyword}' → {match['table']}.{match['column']} = '{match['value']}'")
        
        # Step 2: Generate Candidates
        if verbose:
            print("\nStep 2: Candidate Generation")
            print("-" * 30)
        
        candidates = self._generate_candidates(question, verbose)
        
        if verbose:
            print(f"\nGenerated {len(candidates)} candidates")
        
        # Step 3: Fix Candidates
        if verbose:
            print("\nStep 3: Query Fixing")
            print("-" * 30)
        
        fixed_candidates = self._fix_candidates(candidates, question, verbose)
        
        # Step 4: Select Best Candidate
        if verbose:
            print("\nStep 4: Candidate Selection")
            print("-" * 30)
        
        selection_result = self.selection_agent.select_best_candidate(fixed_candidates, question)
        best_candidate = selection_result['selected_candidate']
        
        if verbose:
            print(f"Selected candidate from {selection_result['selection_details']['reasoning']}")
            print(f"\nFinal SQL: {best_candidate['sql']}")
        
        # Step 5: Execute Final Query
        if verbose:
            print("\nStep 5: Query Execution")
            print("-" * 30)
        
        execution_result = self._execute_query(best_candidate['sql'])
        
        if verbose:
            if execution_result['success']:
                print(f"✓ Query executed successfully")
                print(f"  Returned {execution_result['row_count']} rows")
                if execution_result['results'] and len(execution_result['results']) > 0:
                    print(f"  Sample results: {execution_result['results'][:3]}")
            else:
                print(f"✗ Query execution failed: {execution_result['error']}")
        
        # Update statistics
        if execution_result['success']:
            self.stats['successful_queries'] += 1
            generator = best_candidate.get('reasoning', {}).get('approach', 'unknown')
            if generator in self.stats['generator_successes']:
                self.stats['generator_successes'][generator] += 1
        
        process_time = time.time() - start_time
        
        # Prepare final result
        result = {
            'question': question,
            'sql': best_candidate['sql'],
            'execution_result': execution_result,
            'process_details': {
                'value_retrieval': relevant_values,
                'candidates_generated': len(candidates),
                'candidates_fixed': len([c for c in fixed_candidates if c.get('fixed', False)]),
                'selection_details': selection_result['selection_details'],
                'winning_generator': best_candidate.get('reasoning', {}).get('approach', 'unknown'),
                'process_time': process_time
            },
            'all_candidates': fixed_candidates  # For analysis
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Process completed in {process_time:.2f} seconds")
            print(f"{'='*60}\n")
        
        return result
    
    def _generate_candidates(self, question: str, verbose: bool) -> List[Dict[str, Any]]:
        """Generate candidates using all three generators."""
        
        candidates = []
        schema = self.db.get_schema()
        
        for name, generator in self.generators.items():
            if verbose:
                print(f"  Generating with {name}...", end='')
            
            try:
                candidate = generator.generate_candidate(question, schema)
                candidate['generator'] = name
                candidates.append(candidate)
                
                if verbose:
                    print(f" ✓ (confidence: {candidate.get('confidence', 0):.2f})")
                    
            except Exception as e:
                if verbose:
                    print(f" ✗ (error: {str(e)[:50]}...)")
        
        return candidates
    
    def _fix_candidates(self, candidates: List[Dict[str, Any]], question: str, 
                       verbose: bool) -> List[Dict[str, Any]]:
        """Apply query fixer to candidates that fail initial execution."""
        
        fixed_candidates = []
        
        for candidate in candidates:
            # Test original query
            test_result = self._execute_query(candidate['sql'])
            
            if test_result['success'] and test_result['row_count'] > 0:
                # Query works fine
                candidate['fixed'] = False
                fixed_candidates.append(candidate)
            else:
                # Try to fix
                if verbose:
                    print(f"  Fixing {candidate['generator']} candidate...", end='')
                
                fix_result = self.query_fixer.fix_query(
                    candidate['sql'], 
                    question,
                    test_result.get('error')
                )
                
                if fix_result['success']:
                    candidate['sql'] = fix_result['fixed_query']
                    candidate['fixed'] = True
                    candidate['fix_attempts'] = len(fix_result['attempts'])
                    fixed_candidates.append(candidate)
                    
                    if verbose:
                        print(" ✓")
                else:
                    # Keep original even if fix failed
                    candidate['fixed'] = False
                    candidate['fix_failed'] = True
                    fixed_candidates.append(candidate)
                    
                    if verbose:
                        print(" ✗")
        
        return fixed_candidates
    
    def _execute_query(self, sql: str) -> Dict[str, Any]:
        """Execute a SQL query and return results."""
        try:
            results = self.db.execute_query(sql)
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system performance statistics."""
        
        success_rate = (self.stats['successful_queries'] / self.stats['total_queries'] * 100 
                       if self.stats['total_queries'] > 0 else 0)
        
        generator_stats = {}
        for name, successes in self.stats['generator_successes'].items():
            rate = (successes / self.stats['successful_queries'] * 100 
                   if self.stats['successful_queries'] > 0 else 0)
            generator_stats[name] = {
                'successes': successes,
                'success_rate': rate
            }
        
        return {
            'total_queries': self.stats['total_queries'],
            'successful_queries': self.stats['successful_queries'],
            'success_rate': success_rate,
            'generator_performance': generator_stats
        }
    
    def showcase_examples(self):
        """Run showcase examples demonstrating the system capabilities."""
        
        showcase_questions = [
            # Simple queries
            "How many customers are there?",
            "Show me all products in the Electronics category",
            
            # Moderate complexity
            "What is the average price of products in each category?",
            "Which customers have placed orders?",
            
            # Complex queries
            "What are the top 3 most popular products by quantity sold?",
            "Which customer has the highest average order value?",
            "Show me products with average rating above 4",
            
            # Very complex
            "Which customers from New York have ordered Electronics products and left reviews with rating above 3?"
        ]
        
        print("\n" + "="*80)
        print("CHASE-SQL SHOWCASE")
        print("="*80)
        
        for i, question in enumerate(showcase_questions, 1):
            print(f"\nExample {i}: {question}")
            print("-" * 40)
            
            result = self.process_question(question, verbose=False)
            
            print(f"SQL: {result['sql']}")
            
            if result['execution_result']['success']:
                rows = result['execution_result']['row_count']
                print(f"✓ Success: {rows} rows returned")
                if rows > 0 and result['execution_result']['results']:
                    print(f"  Sample: {result['execution_result']['results'][0]}")
            else:
                print(f"✗ Failed: {result['execution_result']['error']}")
            
            print(f"  Generator: {result['process_details']['winning_generator']}")
            print(f"  Time: {result['process_details']['process_time']:.2f}s")
        
        # Print summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        stats = self.get_statistics()
        print(f"Total queries: {stats['total_queries']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        print("\nGenerator performance:")
        for name, perf in stats['generator_performance'].items():
            print(f"  {name}: {perf['successes']} wins ({perf['success_rate']:.1f}%)")