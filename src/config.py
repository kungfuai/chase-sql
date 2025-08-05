"""
Simple configuration for CHASE-SQL LLM integration.
"""

import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, fallback to os.getenv

class Config:
    """Minimal configuration for LLM features."""
    
    @staticmethod
    def get_gemini_api_key() -> Optional[str]:
        """Get Gemini API key from environment or .env file."""
        return os.getenv('GEMINI_API_KEY')
    
    @staticmethod
    def is_llm_enabled() -> bool:
        """Check if LLM mode is enabled."""
        return Config.get_gemini_api_key() is not None