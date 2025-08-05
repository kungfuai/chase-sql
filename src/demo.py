"""
Demo script showcasing the CHASE-SQL text-to-SQL system.
Run this to see the system in action with various example questions.
"""

import sys
from chase_sql import ChaseSQL

def print_header():
    """Print demo header."""
    print("\n" + "="*80)
    print("CHASE-SQL: Multi-Path Reasoning and Preference Optimized Text-to-SQL")
    print("Based on the paper: https://arxiv.org/pdf/2410.01943")
    print("="*80 + "\n")

def run_interactive_demo():
    """Run interactive demo where users can input their own questions."""
    
    print_header()
    
    # Initialize system
    print("Initializing CHASE-SQL system...")
    system = ChaseSQL()
    print("✓ System ready!\n")
    
    # Show available tables
    print("Database Schema:")
    print("-" * 40)
    print(system.db.get_schema())
    print("\n")
    
    # Interactive loop
    print("Enter natural language questions (or 'quit' to exit, 'examples' for showcase):")
    print("-" * 60)
    
    while True:
        try:
            question = input("\nQuestion: ").strip()
            
            if question.lower() == 'quit':
                break
            
            elif question.lower() == 'examples':
                system.showcase_examples()
                continue
            
            elif question.lower() == 'stats':
                stats = system.get_statistics()
                print("\nSystem Statistics:")
                print(f"  Total queries: {stats['total_queries']}")
                print(f"  Success rate: {stats['success_rate']:.1f}%")
                print("  Generator performance:")
                for name, perf in stats['generator_performance'].items():
                    print(f"    {name}: {perf['successes']} wins ({perf['success_rate']:.1f}%)")
                continue
            
            elif not question:
                continue
            
            # Process the question
            result = system.process_question(question, verbose=True)
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try another question.")

def run_automated_demo():
    """Run automated demo with predefined questions."""
    
    print_header()
    
    # Initialize system
    print("Initializing CHASE-SQL system...")
    system = ChaseSQL()
    print("✓ System ready!\n")
    
    # Define test questions of varying complexity
    test_questions = [
        {
            'question': "How many customers are registered in our system?",
            'category': 'Simple Count',
            'expected_features': ['COUNT']
        },
        {
            'question': "Show me all customers from New York",
            'category': 'Simple Filter',
            'expected_features': ['WHERE', 'city']
        },
        {
            'question': "What are the top 5 most expensive products?",
            'category': 'Sorting and Limiting',
            'expected_features': ['ORDER BY', 'LIMIT', 'price']
        },
        {
            'question': "How many products are in each category?",
            'category': 'Group By Aggregation',
            'expected_features': ['GROUP BY', 'COUNT', 'category']
        },
        {
            'question': "Which customers have spent more than $500 in total?",
            'category': 'Join with Aggregation and Having',
            'expected_features': ['JOIN', 'SUM', 'GROUP BY', 'HAVING']
        },
        {
            'question': "What products have never been ordered?",
            'category': 'Left Join with NULL check',
            'expected_features': ['LEFT JOIN', 'IS NULL']
        },
        {
            'question': "Show me the average rating for each product that has been reviewed",
            'category': 'Join with Aggregation',
            'expected_features': ['JOIN', 'AVG', 'GROUP BY']
        },
        {
            'question': "Which customer placed the most recent order?",
            'category': 'Subquery or Order with Limit',
            'expected_features': ['ORDER BY', 'date', 'LIMIT']
        }
    ]
    
    print(f"Running {len(test_questions)} test questions...\n")
    
    for i, test in enumerate(test_questions, 1):
        print(f"\nTest {i}/{len(test_questions)}: {test['category']}")
        print("=" * 60)
        print(f"Question: {test['question']}")
        print("-" * 60)
        
        # Process question
        result = system.process_question(test['question'], verbose=False)
        
        # Display results
        print(f"\nGenerated SQL:")
        print(f"  {result['sql']}")
        
        print(f"\nExecution Result:")
        if result['execution_result']['success']:
            print(f"  ✓ Success: {result['execution_result']['row_count']} rows")
            if result['execution_result']['results'] and len(result['execution_result']['results']) > 0:
                print(f"  Sample: {result['execution_result']['results'][0]}")
        else:
            print(f"  ✗ Failed: {result['execution_result']['error']}")
        
        print(f"\nProcess Details:")
        print(f"  Winning generator: {result['process_details']['winning_generator']}")
        print(f"  Candidates generated: {result['process_details']['candidates_generated']}")
        print(f"  Candidates fixed: {result['process_details']['candidates_fixed']}")
        print(f"  Process time: {result['process_details']['process_time']:.2f}s")
        
        # Check if expected features are present
        sql_upper = result['sql'].upper()
        features_found = [f for f in test['expected_features'] if f.upper() in sql_upper]
        features_missing = [f for f in test['expected_features'] if f.upper() not in sql_upper]
        
        if features_found:
            print(f"  Expected features found: {', '.join(features_found)}")
        if features_missing:
            print(f"  Expected features missing: {', '.join(features_missing)}")
    
    # Print final statistics
    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)
    
    stats = system.get_statistics()
    print(f"Total queries: {stats['total_queries']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")
    print("\nGenerator performance:")
    for name, perf in stats['generator_performance'].items():
        print(f"  {name}: {perf['successes']} wins ({perf['success_rate']:.1f}%)")

def main():
    """Main entry point for demo."""
    
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        # Run automated demo
        run_automated_demo()
    else:
        # Run interactive demo
        run_interactive_demo()

if __name__ == "__main__":
    main()