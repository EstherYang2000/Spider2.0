from typing import List, Dict, Optional
import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class RefinementLogRAG:
    # Predefined common SQL syntax error correction examples
    PREDEFINED_CASES = [
        {
            "original_sql": "SELECT * FROM users WHERE name = 'O'Reilly'",
            "refined_sql": "SELECT * FROM users WHERE name = 'O''Reilly'",
            "error": "Syntax error: Unterminated string literal",
            "success": True,
            "error_type": "StringLiteral",
            "description": "Fix single quotes inside string literal by doubling them.",
            "dialect": "postgres"
        },
        {
            "original_sql": "SELECT COUNT(*) orders GROUP BY category",
            "refined_sql": "SELECT COUNT(*) FROM orders GROUP BY category",
            "error": "Syntax error: FROM keyword not found",
            "success": True,
            "error_type": "MissingKeyword",
            "description": "FROM clause is required for SELECT statements.",
            "dialect": "sqlite"
        },
        {
            "original_sql": "SELECT * FROM orders JOIN users ON user_id",
            "refined_sql": "SELECT * FROM orders JOIN users ON orders.user_id = users.id",
            "error": "ON clause must be a boolean expression",
            "success": True,
            "error_type": "JoinCondition",
            "description": "JOIN clause must contain a complete boolean expression.",
            "dialect": "postgres"
        },
        {
            "original_sql": "SELECT product, SUM(price) FROM sales",
            "refined_sql": "SELECT product, SUM(price) FROM sales GROUP BY product",
            "error": "Column must appear in the GROUP BY clause",
            "success": True,
            "error_type": "GroupBy",
            "description": "Non-aggregated columns must appear in the GROUP BY clause.",
            "dialect": "postgres"
        },
        {
            "original_sql": "SELECT id, name FROM users JOIN orders ON id = user_id",
            "refined_sql": "SELECT users.id, users.name FROM users JOIN orders ON users.id = orders.user_id",
            "error": "Ambiguous column reference",
            "success": True,
            "error_type": "AmbiguousColumn",
            "description": "Qualify column names with table names to avoid ambiguity.",
            "dialect": "snowflake"
        },
        {
            "original_sql": "SELECT id + '123' FROM users",
            "refined_sql": "SELECT CAST(id AS STRING) || '123' FROM users",
            "error": "Cannot apply '+' to type INT and STRING",
            "success": True,
            "error_type": "TypeMismatch",
            "description": "Use CAST and || for string concatenation.",
            "dialect": "snowflake"
        },
        {
            "original_sql": "SELECT * FROM events WHERE event_date = '2024-01-01'",
            "refined_sql": "SELECT * FROM events WHERE DATE(event_date) = '2024-01-01'",
            "error": "Cannot compare DATE with STRING",
            "success": True,
            "error_type": "DateTime",
            "description": "Ensure both sides of comparison use DATE types.",
            "dialect": "bigquery"
        },
        {
            "original_sql": "SELECT ProductName FROM Orders",
            "refined_sql": "SELECT \"ProductName\" FROM \"Orders\"",
            "error": "Column not found: ProductName",
            "success": True,
            "error_type": "IdentifierCase",
            "description": "Use double quotes for case-sensitive identifiers.",
            "dialect": "snowflake"
        },
        {
            "original_sql": "SELECT user_name FROM customers",
            "refined_sql": "SELECT name FROM customers",
            "error": "no such column: user_name",
            "success": True,
            "error_type": "ColumnNotFound",
            "description": "Check column name spelling or schema.",
            "dialect": "sqlite"
        },
        {
            "original_sql": "SELECT id FROM orders JOIN users ON orders.user_id = users.id",
            "refined_sql": "SELECT orders.id FROM orders JOIN users ON orders.user_id = users.id",
            "error": "column reference 'id' is ambiguous",
            "success": True,
            "error_type": "AmbiguousColumn",
            "description": "Add table name to disambiguate columns.",
            "dialect": "postgres"
        },
        {
            "original_sql": "SELECT MAX(name) WHERE age > 30",
            "refined_sql": "SELECT MAX(name) FROM users WHERE age > 30",
            "error": "FROM clause is missing",
            "success": True,
            "error_type": "MissingKeyword",
            "description": "Always include FROM clause when selecting.",
            "dialect": "sqlite"
        },
        {
            "original_sql": "SELECT * FROM sales GROUP BY product",
            "refined_sql": "SELECT product, COUNT(*) FROM sales GROUP BY product",
            "error": "SELECT list includes non-aggregated columns not in GROUP BY",
            "success": True,
            "error_type": "GroupBy",
            "description": "All SELECT columns must be aggregated or in GROUP BY.",
            "dialect": "bigquery"
        },
        {
            "original_sql": "SELECT * FROM orders WHERE status = completed",
            "refined_sql": "SELECT * FROM orders WHERE status = 'completed'",
            "error": "Unrecognized name: completed",
            "success": True,
            "error_type": "StringLiteral",
            "description": "Use quotes around string literals.",
            "dialect": "bigquery"
        },
        {
            "original_sql": "SELECT * FROM data WHERE value > 10 AND",
            "refined_sql": "SELECT * FROM data WHERE value > 10",
            "error": "Syntax error: incomplete WHERE clause",
            "success": True,
            "error_type": "SyntaxError",
            "description": "Incomplete boolean expression in WHERE clause.",
            "dialect": "sqlite"
        },
        {
            "original_sql": "SELECT TO_DATE('2024-05-01') = created_at FROM logs",
            "refined_sql": "SELECT TO_DATE('2024-05-01') = DATE(created_at) FROM logs",
            "error": "Cannot compare DATE with TIMESTAMP",
            "success": True,
            "error_type": "DateTime",
            "description": "Match comparison types using DATE() or TIMESTAMP().",
            "dialect": "snowflake"
        },
        {
            "original_sql": "SELECT SUM(price) AS total, category FROM products",
            "refined_sql": "SELECT category, SUM(price) AS total FROM products GROUP BY category",
            "error": "Column 'category' must appear in GROUP BY",
            "success": True,
            "error_type": "GroupBy",
            "description": "Reorder SELECT to match GROUP BY requirement.",
            "dialect": "postgres"
        },
        {
            "original_sql": "SELECT COUNT(name, id) FROM users",
            "refined_sql": "SELECT COUNT(*) FROM users",
            "error": "COUNT takes exactly one argument",
            "success": True,
            "error_type": "FunctionNotFound",
            "description": "COUNT must use one argument or *",
            "dialect": "postgres"
        },
        {
            "original_sql": "SELECT * FROM (SELECT name FROM users)",
            "refined_sql": "SELECT name FROM users",
            "error": "Subquery used without alias",
            "success": True,
            "error_type": "SyntaxError",
            "description": "Subqueries must have an alias in most dialects.",
            "dialect": "snowflake"
        },
        {
            "original_sql": "SELECT id, name FROM employees ORDER BY salary DESC LIMIT name",
            "refined_sql": "SELECT id, name FROM employees ORDER BY salary DESC LIMIT 10",
            "error": "LIMIT must be an integer",
            "success": True,
            "error_type": "TypeMismatch",
            "description": "LIMIT clause must use numeric value.",
            "dialect": "bigquery"
        },
        {
            "original_sql": "SELECT AVG('salary') FROM employees",
            "refined_sql": "SELECT AVG(salary) FROM employees",
            "error": "Invalid function argument",
            "success": True,
            "error_type": "TypeMismatch",
            "description": "Aggregate functions must take numeric columns.",
            "dialect": "postgres"
        }
    ]


    def __init__(self, log_path="refinement_history.jsonl", model_name="all-MiniLM-L6-v2", dialect=None):
        self.log_path = log_path
        self.model = SentenceTransformer(model_name)
        self.logs = []
        self.embeddings = None
        self.index = None
        self.dialect = dialect
        self._load_and_embed_logs()

    def _load_and_embed_logs(self):
        self.logs = [case for case in self.PREDEFINED_CASES if case.get("dialect") == self.dialect]

        if os.path.exists(self.log_path):
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        case = json.loads(line)
                        if case.get("dialect") == self.dialect:
                            self.logs.append(case)
                    except Exception:
                        continue

        if not self.logs:
            self.embeddings = None
            return

        texts = [
            f"{case.get('original_sql', '')} || {case.get('refined_sql', '')} || {case.get('error', '')} || {case.get('error_type', '')} || {case.get('description', '')}"
            for case in self.logs
        ]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def retrieve_similar(self, sql_query: str, error_message: str = "", error_type: Optional[str] = None, top_k: int = 2) -> List[Dict[str, any]]:
        self.logs = [case for case in self.PREDEFINED_CASES if case.get("error_type") == error_type]
        if not self.logs or self.embeddings is None:
            return []

        query_text = f"{sql_query} || {error_message} || {error_type or ''}"
        query_emb = self.model.encode([query_text], convert_to_numpy=True)
        _, indices = self.index.search(query_emb, top_k)

        candidates = [self.logs[idx] for idx in indices[0] if idx < len(self.logs)]
        if error_type:
            filtered = [c for c in candidates if c.get("error_type") and error_type.lower() in c["error_type"].lower()]
            return filtered if filtered else candidates
        return candidates



class RefinementMemoryBank:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        log_path: Optional[str] = None,
        base_cases: Optional[List[Dict]] = None,
        filter_key: Optional[str] = None,
        filter_value: Optional[str] = None,
        embed_fields: Optional[List[str]] = None,
    ):
        self.model = SentenceTransformer(model_name)
        self.log_path = log_path
        self.logs: List[Dict] = []
        self.embeddings = None
        self.index = None
        self.embed_fields = embed_fields or ['original_sql', 'refined_sql', 'error_type', 'description']
        self.filter_key = filter_key
        self.filter_value = filter_value

        if base_cases:
            self.logs.extend(self._filter_cases(base_cases))
        if log_path:
            self._load_log_file()
        self._build_index()

    def _filter_cases(self, cases: List[Dict]) -> List[Dict]:
        if self.filter_key and self.filter_value:
            return [c for c in cases if c.get(self.filter_key) == self.filter_value]
        return cases

    def _load_log_file(self):
        if not os.path.exists(self.log_path):
            return
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    case = json.loads(line)
                    self.logs.append(case)
                except Exception:
                    continue
        self.logs = self._filter_cases(self.logs)

    def _build_index(self):
        if not self.logs:
            return
        texts = [
            " || ".join(str(case.get(f, "")) for f in self.embed_fields)
            for case in self.logs
        ]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def retrieve(
        self,
        query_text: str,
        top_k: int = 2,
        constraint_key: Optional[str] = None,
        constraint_value: Optional[str] = None,
    ) -> List[Dict]:
        if self.index is None or self.embeddings is None:
            return []

        query_emb = self.model.encode([query_text], convert_to_numpy=True)
        _, indices = self.index.search(query_emb, top_k)

        candidates = [self.logs[i] for i in indices[0] if i < len(self.logs)]

        if constraint_key and constraint_value:
            filtered = [c for c in candidates if c.get(constraint_key) and constraint_value.lower() in c[constraint_key].lower()]
            return filtered if filtered else candidates
        return candidates


if __name__ == "__main__":
    memory_bank = RefinementMemoryBank(
        model_name="all-MiniLM-L6-v2",
        log_path="refinement_history.jsonl",
        base_cases=RefinementLogRAG.PREDEFINED_CASES,
        filter_key="dialect",
        filter_value="sqlite",
        embed_fields=["original_sql", "refined_sql", "error", "error_type", "description"]
    )

    query_sql = "SELECT * FROM orders WHERE order_date = '2023-01-01'"
    query_error = "Cannot compare DATE with STRING"
    query_type = "DateTime"

    similar_cases = memory_bank.retrieve(
        query_text=f"{query_sql} || {query_error} || {query_type}",
        top_k=3,
        constraint_key="error_type",
        constraint_value=query_type
    )
    print(similar_cases)

    