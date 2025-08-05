"""
Basic test for LLM integration.

Setup:
1. Copy .env.example to .env
2. Add your GEMINI_API_KEY to .env file
3. Run: pip install -r requirements.txt
"""

import os
from chase_sql import ChaseSQL
from config import Config

def test_llm_integration():
    """Test that LLM integration works when enabled."""
    
    # Test config
    has_api_key = Config.get_gemini_api_key() is not None
    print(f"Gemini API key present: {has_api_key}")
    print(f"LLM enabled: {Config.is_llm_enabled()}")
    
    # Test system initialization
    system = ChaseSQL()
    
    print(f"Available generators: {list(system.generators.keys())}")
    
    # Test simple query
    if 'llm' in system.generators:
        print("\nTesting LLM generator...")
        try:
            result = system.process_question("How many customers are there?", verbose=False)
            print(f"LLM query successful: {result['execution_result']['success']}")
            print(f"Generated SQL: {result['sql']}")
            print(f"Winning generator: {result['process_details']['winning_generator']}")
        except Exception as e:
            print(f"LLM test failed: {e}")
    else:
        print("\nLLM generator not available - using rule-based generators only")

if __name__ == "__main__":
    test_llm_integration()