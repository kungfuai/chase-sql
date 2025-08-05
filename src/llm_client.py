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
                self._client = genai.GenerativeModel('gemini-pro')
            except ImportError:
                self._client = None
    
    def is_available(self) -> bool:
        """Check if Gemini client is available."""
        return self._client is not None
    
    def generate_sql(self, question: str, schema: Dict[str, Any]) -> Optional[str]:
        """Generate SQL using Gemini API."""
        if not self.is_available():
            return None
        
        schema_text = self._format_schema(schema)
        prompt = f"""Convert this natural language question to SQL.

Database Schema:
{schema_text}

Question: {question}

Return only the SQL query, no explanation."""
        
        try:
            response = self._client.generate_content(prompt)
            sql = response.text.strip()
            # Clean up response (remove markdown, etc.)
            if sql.startswith('```sql'):
                sql = sql[6:]
            if sql.endswith('```'):
                sql = sql[:-3]
            return sql.strip()
        except Exception:
            return None
    
    def _format_schema(self, schema: Dict[str, Any]) -> str:
        """Format schema for prompt."""
        lines = []
        for table, info in schema.items():
            columns = ", ".join(info['columns'])
            lines.append(f"{table} ({columns})")
        return "\n".join(lines)