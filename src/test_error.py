#!/usr/bin/env python3
"""
Test script to diagnose the LLM generator error.
"""

import sys
import logging
import traceback
from chase_sql import ChaseSQL

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_error.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_question(question):
    """Test a single question."""
    print(f"\n{'='*60}")
    print(f"Testing: {question}")
    print(f"{'='*60}\n")
    
    try:
        system = ChaseSQL()
        result = system.process_question(question, verbose=True)
        
        print(f"\nResult:")
        print(f"  SQL: {result['sql']}")
        print(f"  Success: {result['execution_result']['success']}")
        if result['execution_result']['success']:
            print(f"  Rows: {result['execution_result']['row_count']}")
        else:
            print(f"  Error: {result['execution_result']['error']}")
            
    except Exception as e:
        logger.error(f"Failed to process question: {question}")
        logger.error(f"Error: {str(e)}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        print(f"\nFailed with error: {str(e)}")

if __name__ == "__main__":
    # Test the specific question that caused the error
    test_question("what is the revenue this year")