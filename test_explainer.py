#!/usr/bin/env python3
"""Test script for QueryExplainer functionality."""

import sys
sys.path.append('src')

from chase_sql import ChaseSQL

def test_query_explanation():
    """Test the query explanation feature."""
    
    print("Testing Query Explanation Feature")
    print("=" * 60)
    
    # Initialize system
    system = ChaseSQL()
    
    # Test question
    question = "which product brought in most revenue"
    
    print(f"\nQuestion: {question}")
    print("=" * 60)
    
    # Process question with verbose output
    result = system.process_question(question, verbose=True)
    
    # Check if explanation was generated
    if 'explanation' in result:
        print("\n✓ Query explanation successfully generated!")
        print("\nExplanation details:")
        print(f"  - Annotated SQL included: {'annotated_sql' in result['explanation']}")
        print(f"  - Plain explanation included: {'plain_explanation' in result['explanation']}")
    else:
        print("\n✗ Query explanation not found in result")
    
    return result

if __name__ == "__main__":
    result = test_query_explanation()