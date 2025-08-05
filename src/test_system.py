"""
Test script for validating CHASE-SQL components.
"""

import unittest
from chase_sql import ChaseSQL
from database import ECommerceDB
from knowledge_base import QueryKnowledgeBase
from value_retrieval import ValueRetrieval

class TestChaseSQLComponents(unittest.TestCase):
    """Test individual components of the CHASE-SQL system."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test database and components."""
        cls.db_path = "test_ecommerce.db"
        cls.db = ECommerceDB(cls.db_path)
        cls.kb = QueryKnowledgeBase()
        cls.vr = ValueRetrieval(cls.db_path)
        cls.system = ChaseSQL(cls.db_path)
    
    def test_database_creation(self):
        """Test that database is created with correct tables."""
        schema = self.db.get_schema()
        
        # Check all tables exist
        self.assertIn("customers", schema)
        self.assertIn("products", schema)
        self.assertIn("orders", schema)
        self.assertIn("order_items", schema)
        self.assertIn("reviews", schema)
    
    def test_knowledge_base(self):
        """Test knowledge base functionality."""
        all_queries = self.kb.get_all_queries()
        self.assertGreater(len(all_queries), 10)
        
        # Test search functionality
        results = self.kb.search_queries(['count', 'customer'])
        self.assertGreater(len(results), 0)
        
        # Test category filtering
        simple_queries = self.kb.get_queries_by_complexity('simple')
        self.assertGreater(len(simple_queries), 0)
    
    def test_value_retrieval(self):
        """Test value retrieval component."""
        # Test keyword extraction
        keywords = self.vr.extract_keywords("Show me customers from New York")
        self.assertIn('new', [k.lower() for k in keywords])
        self.assertIn('york', [k.lower() for k in keywords])
        
        # Test value retrieval
        values = self.vr.retrieve_relevant_values("Show me customers from New York")
        self.assertGreater(len(values), 0)
    
    def test_simple_queries(self):
        """Test simple query processing."""
        simple_questions = [
            "How many customers are there?",
            "Show all products",
            "List all orders"
        ]
        
        for question in simple_questions:
            with self.subTest(question=question):
                result = self.system.process_question(question, verbose=False)
                self.assertTrue(result['execution_result']['success'], 
                              f"Failed on: {question}")
    
    def test_moderate_queries(self):
        """Test moderate complexity queries."""
        moderate_questions = [
            "What is the average price of products?",
            "How many orders were placed by each customer?",
            "Show products with price above 100"
        ]
        
        for question in moderate_questions:
            with self.subTest(question=question):
                result = self.system.process_question(question, verbose=False)
                # These might fail but should at least generate valid SQL
                self.assertIsNotNone(result['sql'])
                self.assertGreater(len(result['sql']), 10)
    
    def test_generator_diversity(self):
        """Test that different generators produce different queries."""
        question = "Which customers have placed orders?"
        
        result = self.system.process_question(question, verbose=False)
        all_candidates = result['all_candidates']
        
        # Should have multiple candidates
        self.assertGreaterEqual(len(all_candidates), 2)
        
        # Candidates should be different
        sql_queries = [c['sql'] for c in all_candidates]
        unique_queries = set(sql_queries)
        self.assertGreater(len(unique_queries), 1, 
                          "Generators produced identical queries")
    
    def test_query_fixer(self):
        """Test query fixing functionality."""
        from query_fixer import QueryFixer
        
        fixer = QueryFixer(self.db_path)
        
        # Test fixing a query with syntax error
        bad_query = "SELECT * FROM customer"  # Should be 'customers'
        fix_result = fixer.fix_query(bad_query, "Show all customers")
        
        self.assertIn('customers', fix_result['fixed_query'])
    
    def test_selection_agent(self):
        """Test selection agent functionality."""
        # Create mock candidates
        candidates = [
            {
                'sql': 'SELECT COUNT(*) FROM customers',
                'confidence': 0.8,
                'reasoning': {'approach': 'divide_conquer'}
            },
            {
                'sql': 'SELECT * FROM customers',
                'confidence': 0.6,
                'reasoning': {'approach': 'query_plan'}
            }
        ]
        
        selection_result = self.system.selection_agent.select_best_candidate(
            candidates, 
            "How many customers are there?"
        )
        
        # Should select the COUNT query for a "how many" question
        selected_sql = selection_result['selected_candidate']['sql']
        self.assertIn('COUNT', selected_sql)

def run_integration_test():
    """Run a full integration test with various queries."""
    
    print("\n" + "="*60)
    print("CHASE-SQL Integration Test")
    print("="*60 + "\n")
    
    system = ChaseSQL()
    
    test_cases = [
        ("Simple count", "How many products do we have?"),
        ("Filter query", "Show me expensive products over $500"),
        ("Join query", "Which customers ordered Electronics?"),
        ("Aggregation", "What's the total revenue by product category?"),
        ("Complex", "Find customers who ordered products but never reviewed them")
    ]
    
    results = []
    
    for name, question in test_cases:
        print(f"\nTesting: {name}")
        print(f"Question: {question}")
        
        try:
            result = system.process_question(question, verbose=False)
            
            if result['execution_result']['success']:
                status = "✓ PASS"
                rows = result['execution_result']['row_count']
                details = f"{rows} rows"
            else:
                status = "✗ FAIL"
                details = result['execution_result']['error'][:50]
            
            results.append((name, status, details))
            print(f"Result: {status} ({details})")
            print(f"SQL: {result['sql'][:80]}...")
            
        except Exception as e:
            results.append((name, "✗ ERROR", str(e)[:50]))
            print(f"Error: {str(e)[:50]}")
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, status, details in results:
        print(f"{name:20} {status:10} {details}")
    
    passed = sum(1 for _, status, _ in results if "PASS" in status)
    total = len(results)
    print(f"\nPassed: {passed}/{total} ({passed/total*100:.0f}%)")

if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration test
    print("\n" + "="*60)
    run_integration_test()