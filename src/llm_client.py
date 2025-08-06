"""
Lightweight Gemini API client for CHASE-SQL.
"""

from typing import Optional, Dict, Any
from config import Config

class GeminiClient:
    """Simple Gemini API client."""
    
    def __init__(self):
        self.api_key = Config.get_gemini_api_key()
        self._client = None
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel('gemini-1.5-flash')
            except ImportError:
                self._client = None
    
    def is_available(self) -> bool:
        """Check if Gemini client is available."""
        return self._client is not None
    
    def generate_sql(self, question: str, schema: Dict[str, Any]) -> Optional[str]:
        """Generate SQL using Gemini API."""
        import logging
        logger = logging.getLogger(__name__)
        
        if not self.is_available():
            logger.error("LLM client not available")
            return None
        
        try:
            schema_text = self._format_schema(schema)
            logger.debug(f"Formatted schema: {schema_text}")
        except Exception as e:
            logger.error(f"Failed to format schema: {str(e)}")
            logger.error(f"Schema received: {schema}")
            return None
        
        prompt = f"""Convert this natural language question to SQL.

Database Schema:
{schema_text}

Question: {question}

Return only the SQL query, no explanation."""
        
        try:
            logger.debug("Calling Gemini API...")
            response = self._client.generate_content(prompt)
            sql = response.text.strip()
            logger.debug(f"Raw response: {sql}")
            
            # Clean up response (remove markdown, etc.)
            if sql.startswith('```sql'):
                sql = sql[6:]
            if sql.endswith('```'):
                sql = sql[:-3]
            
            result = sql.strip()
            logger.debug(f"Cleaned SQL: {result}")
            return result
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            import traceback
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            return None
    
    def generate(self, prompt: str) -> Optional[str]:
        """Generate text using Gemini API."""
        if not self.is_available():
            return None
        
        try:
            response = self._client.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Gemini API call failed: {str(e)}")
            return None
    
    def _format_schema(self, schema: Dict[str, Any]) -> str:
        """Format schema for prompt."""
        lines = []
        for table, info in schema.items():
            columns = ", ".join(info['columns'])
            lines.append(f"{table} ({columns})")
        return "\n".join(lines)