# CHASE-SQL: Multi-Path Reasoning and Preference Optimized Text-to-SQL

A working implementation of the CHASE-SQL system based on the paper ["CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection in Text-to-SQL"](https://arxiv.org/pdf/2410.01943) (Pourreza et al., 2024).

## Overview

CHASE-SQL is an advanced text-to-SQL system that employs innovative strategies to convert natural language questions into SQL queries. It uses **multi-path reasoning** to generate diverse SQL candidates and **preference-optimized selection** to identify the best query.

### Key Innovations

1. **Multi-Agent Approach**: Three specialized generators create diverse SQL candidates
2. **Intelligent Selection**: Pairwise comparison identifies the optimal query
3. **Self-Reflection**: Automatic error correction improves success rates
4. **Value Retrieval**: LSH-based matching connects natural language to database values

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Natural Language│     │ Database Schema  │     │ Retrieved Values│
│    Question     │     │   Information    │     │   (LSH-based)   │
└────────┬────────┘     └────────┬─────────┘     └────────┬────────┘
         │                       │                          │
         └───────────────────────┴──────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Candidate Generators  │
                    ├─────────────────────────┤
                    │ • Divide & Conquer CoT │
                    │ • Query Plan CoT       │
                    │ • Synthetic Examples   │
                    └────────────┬────────────┘
                                 │
                         ┌───────▼────────┐
                         │  Query Fixer   │
                         │ (Self-Reflect) │
                         └───────┬────────┘
                                 │
                       ┌─────────▼─────────┐
                       │ Selection Agent   │
                       │ (Pairwise Comp.)  │
                       └─────────┬─────────┘
                                 │
                           ┌─────▼─────┐
                           │ Final SQL │
                           └───────────┘
```

## 📁 Project Structure

```
chase_sql/
├── doc/
│   └── 2410.01943v1.pdf    # Original CHASE-SQL paper
├── src/
│   ├── database.py          # Mock e-commerce database with sample data
│   ├── knowledge_base.py    # Pre-defined query templates and examples
│   ├── value_retrieval.py   # LSH-based value matching system
│   ├── generators.py        # Three SQL candidate generators
│   ├── query_fixer.py       # Self-reflection based query correction
│   ├── selection_agent.py   # Pairwise comparison selection mechanism
│   ├── chase_sql.py         # Main system orchestration
│   ├── demo.py             # Interactive and automated demos
│   └── test_system.py      # Comprehensive test suite
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- SQLite3 (included in Python standard library)
- NumPy

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd chase_sql

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy
```

### Run the Interactive Demo

```bash
cd src
python demo.py
```

You'll see:
```
============================================================
CHASE-SQL: Multi-Path Reasoning and Preference Optimized Text-to-SQL
Based on the paper: https://arxiv.org/pdf/2410.01943
============================================================

Initializing CHASE-SQL system...
✓ System ready!

Enter natural language questions (or 'quit' to exit, 'examples' for showcase):
------------------------------------------------------------

Question: How many customers are from New York?
```

### Run Automated Tests

```bash
cd src
python demo.py --auto  # Run pre-defined test cases
python test_system.py  # Run unit tests
```

## 💡 Example Usage

### Basic Usage

```python
from chase_sql import ChaseSQL

# Initialize the system
system = ChaseSQL()

# Process a natural language question
result = system.process_question("What are the top 5 most expensive products?")

# Access the results
print(f"Generated SQL: {result['sql']}")
print(f"Execution successful: {result['execution_result']['success']}")
print(f"Number of rows returned: {result['execution_result']['row_count']}")
print(f"Sample results: {result['execution_result']['results'][:3]}")
```

### Example Questions and Generated SQL

| Natural Language Question | Generated SQL |
|--------------------------|---------------|
| "How many customers are there?" | `SELECT COUNT(*) FROM customers` |
| "Show me products under $50" | `SELECT * FROM products WHERE price < 50` |
| "What's the average order value by city?" | `SELECT c.city, AVG(o.total_amount) FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.city` |
| "Which products have never been ordered?" | `SELECT p.name FROM products p LEFT JOIN order_items oi ON p.id = oi.product_id WHERE oi.product_id IS NULL` |
| "Top 3 customers by total spending" | `SELECT c.name, SUM(o.total_amount) as total FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id ORDER BY total DESC LIMIT 3` |

## 🔧 Core Components

### 1. Value Retrieval System

Extracts keywords from questions and finds matching database values using Locality-Sensitive Hashing (LSH):

```python
# Example: "Show customers from New York"
# Extracts: ["customers", "New", "York"]
# Matches: customers.city = "New York"
```

### 2. Candidate Generators

#### Divide & Conquer CoT
- Breaks complex questions into sub-problems
- Best for: Multi-condition queries, nested logic
- Example: "Find customers who ordered electronics and spent over $500"

#### Query Plan CoT
- Mirrors database execution steps
- Best for: JOIN-heavy queries, systematic table scanning
- Example: "Show all orders with customer and product details"

#### Online Synthetic Examples
- Generates relevant few-shot examples on-the-fly
- Best for: General queries, pattern matching
- Example: "What's the average rating for each category?"

### 3. Query Fixer

Automatically corrects common SQL errors:
- Syntax errors (missing commas, quotes)
- Invalid table/column names
- Ambiguous column references
- Empty result handling

### 4. Selection Agent

Uses pairwise comparison to select the best candidate:
- Compares execution success
- Evaluates result quality
- Considers query-question alignment
- Tournament-style scoring


## 🗄️ Database Schema

The system uses a mock e-commerce database:

```sql
customers (id, name, email, city, registration_date)
products (id, name, category, price, stock_quantity)
orders (id, customer_id, order_date, total_amount, status)
order_items (id, order_id, product_id, quantity, unit_price)
reviews (id, product_id, customer_id, rating, review_text, date)
```

### Stats About Mock Dataset

**Schema Complexity: MODERATE**
- 5 interconnected tables with proper foreign key relationships
- Mix of data types (INTEGER, TEXT, DECIMAL, DATE)
- Built-in constraints and referential integrity

**Data Scale: LOW-MODERATE**
- 10 customers, 15 products (4 categories)
- 50 orders with 1-5 items each
- 100 product reviews

**Query Complexity Support:**
- ✅ **Basic**: Simple SELECT, WHERE, COUNT queries
- ✅ **Intermediate**: JOINs, GROUP BY, aggregations (SUM, AVG, etc.)
- ✅ **Advanced**: Multi-table JOINs, subqueries, HAVING clauses
- ❌ **Complex**: Window functions, CTEs, recursive queries
- ❌ **Performance Testing**: Limited by small dataset size

**Supported Query Patterns:**
- Simple filtering and counting
- Category-based aggregations
- Customer spending analysis
- Product popularity and ratings
- Order history and status tracking
- Cross-table relationship queries (customers ↔ orders ↔ products)

## 🧪 Testing

### Unit Tests
```bash
python test_system.py
```

Tests cover:
- Database creation and population
- Value retrieval accuracy
- Generator diversity
- Query fixer capabilities
- Selection agent logic

### Integration Tests
The automated demo includes various query categories:
- Simple aggregations
- Filtering queries
- JOIN operations
- Complex aggregations
- Multi-table queries

## 🔍 Advanced Features

### Custom Questions
The system handles various SQL patterns:
- Aggregations (COUNT, SUM, AVG, MIN, MAX)
- Filtering (WHERE clauses with multiple conditions)
- Joins (INNER, LEFT, multiple tables)
- Grouping (GROUP BY with HAVING)
- Ordering and limiting (ORDER BY, LIMIT)
- Date operations (date filtering, formatting)

### Debugging and Verbosity
```python
# Verbose mode shows all steps
result = system.process_question("Your question here", verbose=True)
```

This displays:
- Value retrieval results
- Each generator's output
- Query fixing attempts
- Selection process
- Final execution details

## 📈 Limitations and Future Enhancements

### Current Limitations
1. **Rule-based Generation**: Uses heuristics instead of neural models
2. **Limited SQL Features**: No support for CTEs, window functions, or complex subqueries
3. **Static Selection**: Heuristic-based rather than learned selection

### Potential Improvements
1. **LLM Integration**: Replace rule-based generators with GPT-4/Claude
2. **Fine-tuned Selection**: Train selection model on real data
3. **Extended SQL Support**: Add more complex SQL features
4. **Better Schema Linking**: Implement neural schema linking
5. **Multi-database Support**: Extend beyond SQLite

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional generator strategies
- More sophisticated value retrieval
- Enhanced query fixing rules
- Better selection heuristics
- Extended test coverage

## 📚 Citation

If you use this implementation in your research, please cite:

```bibtex
@article{pourreza2024chase,
  title={CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection in Text-to-SQL},
  author={Pourreza, Mohammadreza and Li, Hailong and Sun, Ruoxi and Chung, Yeounoh and 
          Talaei, Shayan and Kakkar, Gaurav Tarlok and Gan, Yu and Saberi, Amin and 
          Özcan, Fatma and Arık, Sercan Ö.},
  journal={arXiv preprint arXiv:2410.01943},
  year={2024}
}
```

## 📄 License

This implementation is provided for educational and research purposes. Please refer to the original paper for more details about the CHASE-SQL system.

## 🙏 Acknowledgments

This implementation is based on the innovative work presented in the CHASE-SQL paper by the Google Cloud and Stanford University research teams. The implementation demonstrates the core concepts while using simplified rule-based approaches instead of neural models.
