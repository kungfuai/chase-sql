"""
Database management module for CHASE-SQL implementation.
Creates and manages the mock e-commerce database with sample data.
"""

import sqlite3
import random
from datetime import datetime, timedelta
from typing import List, Tuple

class ECommerceDB:
    def __init__(self, db_path: str = "ecommerce.db"):
        self.db_path = db_path
        self.conn = None
        self.setup_database()
    
    def setup_database(self):
        """Initialize database connection and create tables."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.create_tables()
        self.populate_sample_data()
    
    def create_tables(self):
        """Create all database tables."""
        
        # Customers table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                city TEXT NOT NULL,
                registration_date DATE NOT NULL
            )
        """)
        
        # Products table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price DECIMAL(10,2) NOT NULL,
                stock_quantity INTEGER NOT NULL
            )
        """)
        
        # Orders table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL,
                order_date DATE NOT NULL,
                total_amount DECIMAL(10,2) NOT NULL,
                status TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(id)
            )
        """)
        
        # Order items table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS order_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                unit_price DECIMAL(10,2) NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders(id),
                FOREIGN KEY (product_id) REFERENCES products(id)
            )
        """)
        
        # Reviews table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER NOT NULL,
                customer_id INTEGER NOT NULL,
                rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                review_text TEXT,
                date DATE NOT NULL,
                FOREIGN KEY (product_id) REFERENCES products(id),
                FOREIGN KEY (customer_id) REFERENCES customers(id)
            )
        """)
        
        self.conn.commit()
    
    def populate_sample_data(self):
        """Populate tables with realistic sample data."""
        
        # Sample customers
        customers = [
            ("Alice Johnson", "alice@email.com", "New York", "2023-01-15"),
            ("Bob Smith", "bob@email.com", "Los Angeles", "2023-02-20"),
            ("Carol Davis", "carol@email.com", "Chicago", "2023-01-30"),
            ("David Wilson", "david@email.com", "Houston", "2023-03-10"),
            ("Eve Brown", "eve@email.com", "Phoenix", "2023-02-14"),
            ("Frank Miller", "frank@email.com", "Philadelphia", "2023-01-25"),
            ("Grace Lee", "grace@email.com", "San Antonio", "2023-03-05"),
            ("Henry Taylor", "henry@email.com", "San Diego", "2023-02-28"),
            ("Ivy Chen", "ivy@email.com", "Dallas", "2023-01-18"),
            ("Jack Anderson", "jack@email.com", "San Jose", "2023-03-12")
        ]
        
        self.conn.executemany(
            "INSERT OR IGNORE INTO customers (name, email, city, registration_date) VALUES (?, ?, ?, ?)",
            customers
        )
        
        # Sample products
        products = [
            ("iPhone 15", "Electronics", 999.99, 50),
            ("Samsung Galaxy S24", "Electronics", 899.99, 30),
            ("MacBook Pro", "Electronics", 1999.99, 20),
            ("Dell XPS 13", "Electronics", 1299.99, 25),
            ("Sony WH-1000XM5", "Electronics", 349.99, 100),
            ("Nike Air Max", "Clothing", 129.99, 75),
            ("Adidas Ultraboost", "Clothing", 179.99, 60),
            ("Levi's 501 Jeans", "Clothing", 69.99, 80),
            ("Patagonia Jacket", "Clothing", 249.99, 40),
            ("The Great Gatsby", "Books", 12.99, 200),
            ("Python Programming", "Books", 39.99, 150),
            ("Coffee Maker", "Home", 89.99, 45),
            ("Instant Pot", "Home", 119.99, 35),
            ("Robot Vacuum", "Home", 299.99, 25),
            ("Air Purifier", "Home", 199.99, 30)
        ]
        
        self.conn.executemany(
            "INSERT OR IGNORE INTO products (name, category, price, stock_quantity) VALUES (?, ?, ?, ?)",
            products
        )
        
        # Generate sample orders and order items
        self._generate_orders_and_items()
        
        # Generate sample reviews
        self._generate_reviews()
        
        self.conn.commit()
    
    def _generate_orders_and_items(self):
        """Generate realistic orders and order items."""
        order_statuses = ["pending", "shipped", "delivered", "cancelled"]
        
        # Generate 50 orders
        for i in range(50):
            customer_id = random.randint(1, 10)
            days_ago = random.randint(1, 90)
            order_date = (datetime.now() - timedelta(days=days_ago)).date()
            status = random.choice(order_statuses)
            
            # Create order first to get order_id
            cursor = self.conn.execute(
                "INSERT INTO orders (customer_id, order_date, total_amount, status) VALUES (?, ?, ?, ?)",
                (customer_id, order_date, 0.0, status)  # Will update total_amount later
            )
            order_id = cursor.lastrowid
            
            # Add 1-5 items to each order
            total_amount = 0.0
            num_items = random.randint(1, 5)
            
            for _ in range(num_items):
                product_id = random.randint(1, 15)
                quantity = random.randint(1, 3)
                
                # Get product price
                price_result = self.conn.execute(
                    "SELECT price FROM products WHERE id = ?", (product_id,)
                ).fetchone()
                unit_price = price_result[0] if price_result else 0.0
                
                self.conn.execute(
                    "INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (?, ?, ?, ?)",
                    (order_id, product_id, quantity, unit_price)
                )
                
                total_amount += unit_price * quantity
            
            # Update order total
            self.conn.execute(
                "UPDATE orders SET total_amount = ? WHERE id = ?",
                (total_amount, order_id)
            )
    
    def _generate_reviews(self):
        """Generate sample product reviews."""
        review_texts = [
            "Great product, highly recommended!",
            "Good value for money.",
            "Could be better, but decent quality.",
            "Excellent! Exceeded my expectations.",
            "Not bad, but there are better alternatives.",
            "Amazing quality and fast shipping!",
            "Perfect for my needs.",
            "Would buy again.",
            "Disappointing, expected more.",
            "Outstanding product!"
        ]
        
        # Generate 100 reviews
        for _ in range(100):
            product_id = random.randint(1, 15)
            customer_id = random.randint(1, 10)
            rating = random.randint(1, 5)
            review_text = random.choice(review_texts)
            days_ago = random.randint(1, 60)
            date = (datetime.now() - timedelta(days=days_ago)).date()
            
            try:
                self.conn.execute(
                    "INSERT INTO reviews (product_id, customer_id, rating, review_text, date) VALUES (?, ?, ?, ?, ?)",
                    (product_id, customer_id, rating, review_text, date)
                )
            except sqlite3.IntegrityError:
                # Skip duplicate reviews from same customer for same product
                continue
    
    def execute_query(self, query: str) -> List[Tuple]:
        """Execute a SQL query and return results."""
        try:
            cursor = self.conn.execute(query)
            return cursor.fetchall()
        except Exception as e:
            raise Exception(f"Query execution error: {str(e)}")
    
    def get_schema(self) -> str:
        """Get database schema information."""
        schema_info = []
        
        # Get table names
        tables = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        
        for table in tables:
            table_name = table[0]
            schema_info.append(f"\nTable: {table_name}")
            
            # Get column information
            columns = self.conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            for col in columns:
                col_name, col_type = col[1], col[2]
                schema_info.append(f"  {col_name}: {col_type}")
        
        return "\n".join(schema_info)
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()