"""
Value retrieval module implementing LSH-based keyword extraction and value matching.
Based on the CHASE-SQL paper's approach to retrieving relevant database values.
"""

import re
import sqlite3
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import hashlib
import random

class ValueRetrieval:
    """
    Implements locality-sensitive hashing (LSH) for retrieving relevant database values
    based on keywords extracted from natural language questions.
    """
    
    def __init__(self, database_path: str):
        self.db_path = database_path
        self.value_cache = {}  # Cache for database values
        self.lsh_buckets = defaultdict(set)  # LSH hash buckets
        self.keyword_patterns = [
            r'\b[A-Z][a-z]+\b',  # Capitalized words (likely proper nouns)
            r'\b\d+\.?\d*\b',    # Numbers
            r'\b[a-z]+@[a-z]+\.[a-z]+\b',  # Email patterns
            r'\b\w{2,}\b'        # General words with 2+ characters
        ]
        self._build_value_index()
    
    def _build_value_index(self):
        """Build an index of all string values in the database for LSH matching."""
        conn = sqlite3.connect(self.db_path)
        
        # Get all tables
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        
        for table in tables:
            table_name = table[0]
            
            # Get column info
            columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            text_columns = [col[1] for col in columns if 'TEXT' in col[2].upper()]
            
            if text_columns:
                # Sample values from text columns
                for col in text_columns:
                    try:
                        values = conn.execute(
                            f"SELECT DISTINCT {col} FROM {table_name} WHERE {col} IS NOT NULL LIMIT 100"
                        ).fetchall()
                        
                        for value_tuple in values:
                            value = str(value_tuple[0])
                            if len(value) > 1:  # Filter out single characters
                                self.value_cache[value.lower()] = {
                                    'original': value,
                                    'table': table_name,
                                    'column': col
                                }
                                self._add_to_lsh(value.lower())
                    except Exception:
                        continue
        
        conn.close()
    
    def _add_to_lsh(self, value: str, num_hashes: int = 10):
        """Add a value to LSH buckets using multiple hash functions."""
        for i in range(num_hashes):
            # Create different hash functions by varying the seed
            hash_val = self._hash_function(value, seed=i)
            self.lsh_buckets[hash_val].add(value)
    
    def _hash_function(self, value: str, seed: int = 0) -> str:
        """Simple hash function for LSH."""
        # Create n-grams and hash them
        ngrams = self._get_ngrams(value, n=3)
        combined = ''.join(sorted(ngrams)) + str(seed)
        return hashlib.md5(combined.encode()).hexdigest()[:8]
    
    def _get_ngrams(self, text: str, n: int = 3) -> List[str]:
        """Generate character n-grams from text."""
        text = text.lower().replace(' ', '')
        return [text[i:i+n] for i in range(len(text) - n + 1)]
    
    def extract_keywords(self, question: str) -> List[str]:
        """
        Extract potential keywords from a natural language question.
        Uses pattern matching to identify likely database values.
        """
        keywords = set()
        
        # Apply different patterns to extract keywords
        for pattern in self.keyword_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            keywords.update(matches)
        
        # Clean and filter keywords
        cleaned_keywords = []
        stopwords = {'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                    'of', 'with', 'by', 'what', 'which', 'who', 'how', 'many', 'much', 
                    'where', 'when', 'why', 'all', 'any', 'some', 'each', 'every',
                    'from', 'have', 'has', 'do', 'does', 'did', 'will', 'would', 'can', 'could'}
        
        for keyword in keywords:
            keyword_clean = keyword.lower().strip()
            if len(keyword_clean) > 1 and keyword_clean not in stopwords:
                cleaned_keywords.append(keyword_clean)
        
        return cleaned_keywords
    
    def find_similar_values(self, keyword: str, max_results: int = 5) -> List[Dict]:
        """
        Find database values similar to a keyword using LSH.
        Returns matches with similarity scores and database location info.
        """
        keyword_lower = keyword.lower()
        similar_values = []
        
        # Direct exact match
        if keyword_lower in self.value_cache:
            similar_values.append({
                'value': self.value_cache[keyword_lower]['original'],
                'table': self.value_cache[keyword_lower]['table'],
                'column': self.value_cache[keyword_lower]['column'],
                'similarity': 1.0,
                'match_type': 'exact'
            })
        
        # LSH-based similarity search
        candidates = set()
        
        # Get candidates from LSH buckets
        for i in range(10):  # Same number of hash functions as in _add_to_lsh
            hash_val = self._hash_function(keyword_lower, seed=i)
            if hash_val in self.lsh_buckets:
                candidates.update(self.lsh_buckets[hash_val])
        
        # Score candidates by similarity
        for candidate in candidates:
            if candidate != keyword_lower:  # Skip exact matches (already added)
                similarity = self._calculate_similarity(keyword_lower, candidate)
                if similarity > 0.3:  # Threshold for similarity
                    if candidate in self.value_cache:
                        similar_values.append({
                            'value': self.value_cache[candidate]['original'],
                            'table': self.value_cache[candidate]['table'],
                            'column': self.value_cache[candidate]['column'],
                            'similarity': similarity,
                            'match_type': 'similar'
                        })
        
        # Sort by similarity and return top results
        similar_values.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_values[:max_results]
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings using Jaccard similarity of n-grams."""
        ngrams1 = set(self._get_ngrams(str1, n=3))
        ngrams2 = set(self._get_ngrams(str2, n=3))
        
        if not ngrams1 and not ngrams2:
            return 1.0
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
    
    def retrieve_relevant_values(self, question: str) -> Dict[str, List[Dict]]:
        """
        Main method to retrieve relevant values for a question.
        Returns a mapping of keywords to their similar database values.
        """
        keywords = self.extract_keywords(question)
        relevant_values = {}
        
        for keyword in keywords:
            similar_values = self.find_similar_values(keyword)
            if similar_values:
                relevant_values[keyword] = similar_values
        
        return relevant_values
    
    def get_value_context(self, table: str, column: str, value: str) -> str:
        """Get context information about how a value is used in the database."""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get some example rows containing this value
            examples = conn.execute(
                f"SELECT * FROM {table} WHERE {column} = ? LIMIT 3",
                (value,)
            ).fetchall()
            
            if examples:
                return f"Found in {table}.{column}, appears in {len(examples)} example(s)"
            else:
                return f"Value found in {table}.{column}"
        except Exception as e:
            return f"Value in {table}.{column} (context unavailable: {str(e)})"
        finally:
            conn.close()
    
    def suggest_where_clauses(self, question: str) -> List[str]:
        """Suggest potential WHERE clause conditions based on retrieved values."""
        relevant_values = self.retrieve_relevant_values(question)
        suggestions = []
        
        for keyword, matches in relevant_values.items():
            for match in matches[:2]:  # Top 2 matches per keyword
                table = match['table']
                column = match['column']
                value = match['value']
                
                # Generate different types of conditions
                suggestions.extend([
                    f"{table}.{column} = '{value}'",
                    f"{table}.{column} LIKE '%{value}%'",
                    f"LOWER({table}.{column}) = LOWER('{value}')"
                ])
        
        return suggestions[:10]  # Return top 10 suggestions