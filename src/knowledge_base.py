"""
Knowledge base module containing sample queries and their natural language descriptions.
These serve as the working knowledge base for the CHASE-SQL system.
"""

from typing import List, Dict, Tuple

class QueryKnowledgeBase:
    """Contains sample queries that work with the e-commerce database."""
    
    def __init__(self):
        self.queries = self._initialize_queries()
    
    def _initialize_queries(self) -> List[Dict]:
        """Initialize the knowledge base with sample query-description pairs."""
        
        return [
            {
                "description": "How many customers are registered in the system?",
                "sql": "SELECT COUNT(*) FROM customers",
                "category": "simple_count",
                "complexity": "simple"
            },
            {
                "description": "List all customers from New York",
                "sql": "SELECT name, email FROM customers WHERE city = 'New York'",
                "category": "simple_filter",
                "complexity": "simple"
            },
            {
                "description": "What are the most expensive products?",
                "sql": "SELECT name, price FROM products ORDER BY price DESC LIMIT 5",
                "category": "simple_sort",
                "complexity": "simple"
            },
            {
                "description": "How many products are in each category?",
                "sql": "SELECT category, COUNT(*) as product_count FROM products GROUP BY category",
                "category": "group_by",
                "complexity": "moderate"
            },
            {
                "description": "What is the average price of products in each category?",
                "sql": "SELECT category, AVG(price) as avg_price FROM products GROUP BY category",
                "category": "aggregation",
                "complexity": "moderate"
            },
            {
                "description": "Which customers have placed orders?",
                "sql": "SELECT DISTINCT c.name, c.email FROM customers c JOIN orders o ON c.id = o.customer_id",
                "category": "join",
                "complexity": "moderate"
            },
            {
                "description": "What is the total amount spent by each customer?",
                "sql": "SELECT c.name, SUM(o.total_amount) as total_spent FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name",
                "category": "join_aggregation",
                "complexity": "moderate"
            },
            {
                "description": "Which products have never been ordered?",
                "sql": "SELECT p.name FROM products p LEFT JOIN order_items oi ON p.id = oi.product_id WHERE oi.product_id IS NULL",
                "category": "left_join",
                "complexity": "moderate"
            },
            {
                "description": "What are the top 3 most popular products by quantity sold?",
                "sql": "SELECT p.name, SUM(oi.quantity) as total_sold FROM products p JOIN order_items oi ON p.id = oi.product_id GROUP BY p.id, p.name ORDER BY total_sold DESC LIMIT 3",
                "category": "join_aggregation_sort",
                "complexity": "complex"
            },
            {
                "description": "Which customer has the highest average order value?",
                "sql": "SELECT c.name, AVG(o.total_amount) as avg_order_value FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name ORDER BY avg_order_value DESC LIMIT 1",
                "category": "join_aggregation_sort",
                "complexity": "complex"
            },
            {
                "description": "What products have an average rating above 4?",
                "sql": "SELECT p.name, AVG(r.rating) as avg_rating FROM products p JOIN reviews r ON p.id = r.product_id GROUP BY p.id, p.name HAVING AVG(r.rating) > 4",
                "category": "join_having",
                "complexity": "complex"
            },
            {
                "description": "How many orders were placed each month?",
                "sql": "SELECT strftime('%Y-%m', order_date) as month, COUNT(*) as order_count FROM orders GROUP BY month ORDER BY month",
                "category": "date_aggregation",
                "complexity": "moderate"
            },
            {
                "description": "Which products in Electronics category cost more than the average price?",
                "sql": "SELECT name, price FROM products WHERE category = 'Electronics' AND price > (SELECT AVG(price) FROM products WHERE category = 'Electronics')",
                "category": "subquery",
                "complexity": "complex"
            },
            {
                "description": "What is the revenue generated from each product category?",
                "sql": "SELECT p.category, SUM(oi.quantity * oi.unit_price) as revenue FROM products p JOIN order_items oi ON p.id = oi.product_id GROUP BY p.category ORDER BY revenue DESC",
                "category": "join_aggregation_sort",
                "complexity": "complex"
            },
            {
                "description": "Which customers have written reviews for products they purchased?",
                "sql": "SELECT DISTINCT c.name FROM customers c JOIN orders o ON c.id = o.customer_id JOIN order_items oi ON o.id = oi.order_id JOIN reviews r ON c.id = r.customer_id AND oi.product_id = r.product_id",
                "category": "multiple_joins",
                "complexity": "complex"
            },
            {
                "description": "What is the most recent order for each customer?",
                "sql": "SELECT c.name, MAX(o.order_date) as last_order_date, o.total_amount FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name",
                "category": "join_aggregation",
                "complexity": "moderate"
            },
            {
                "description": "How many customers registered each month this year?",
                "sql": "SELECT strftime('%Y-%m', registration_date) as month, COUNT(*) as new_customers FROM customers WHERE registration_date >= '2023-01-01' GROUP BY month ORDER BY month",
                "category": "date_filter_aggregation",
                "complexity": "moderate"
            },
            {
                "description": "Which products have the highest revenue per unit?",
                "sql": "SELECT p.name, (SUM(oi.quantity * oi.unit_price) / SUM(oi.quantity)) as revenue_per_unit FROM products p JOIN order_items oi ON p.id = oi.product_id GROUP BY p.id, p.name ORDER BY revenue_per_unit DESC LIMIT 5",
                "category": "calculated_field",
                "complexity": "complex"
            },
            {
                "description": "What percentage of orders are delivered vs other statuses?",
                "sql": "SELECT status, COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders) as percentage FROM orders GROUP BY status",
                "category": "percentage_calculation",
                "complexity": "complex"
            },
            {
                "description": "Which city has customers who spend the most on average?",
                "sql": "SELECT c.city, AVG(o.total_amount) as avg_spending FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.city ORDER BY avg_spending DESC LIMIT 1",
                "category": "join_aggregation_sort",
                "complexity": "complex"
            }
        ]
    
    def get_all_queries(self) -> List[Dict]:
        """Get all queries in the knowledge base."""
        return self.queries
    
    def get_queries_by_category(self, category: str) -> List[Dict]:
        """Get queries filtered by category."""
        return [q for q in self.queries if q["category"] == category]
    
    def get_queries_by_complexity(self, complexity: str) -> List[Dict]:
        """Get queries filtered by complexity level."""
        return [q for q in self.queries if q["complexity"] == complexity]
    
    def search_queries(self, keywords: List[str]) -> List[Dict]:
        """Search queries by keywords in description."""
        matching_queries = []
        keywords_lower = [kw.lower() for kw in keywords]
        
        for query in self.queries:
            description_lower = query["description"].lower()
            if any(kw in description_lower for kw in keywords_lower):
                matching_queries.append(query)
        
        return matching_queries
    
    def get_categories(self) -> List[str]:
        """Get all unique categories."""
        return list(set(q["category"] for q in self.queries))
    
    def get_sample_for_synthetic_generation(self, count: int = 5) -> List[Dict]:
        """Get sample queries for synthetic example generation."""
        import random
        return random.sample(self.queries, min(count, len(self.queries)))