"""
Refinement Strategies Module

This module defines comprehensive SQL refinement strategies for different types of database errors.
It provides a collection of actionable suggestions for fixing common SQL issues and a selector
function to format these strategies into prompts for language models to use in query refinement.
"""

REFINEMENT_STRATEGIES = {
    "SyntaxError": [
        "Fix misplaced commas, parentheses, and clause ordering.",
        "Remove or replace invalid keywords or operators.",
        "Simplify nested queries or CASE statements.",
        "Check SELECT/FROM/WHERE/GROUP BY clause order and completeness.",
    ],
    "ColumnNotFound": [
        "Replace the missing or invalid column with one that exists in the schema.",
        "Verify correct table alias or add the table prefix to the column.",
        "Check for typos or casing issues in column names.",
        "Adjust JOINs to ensure the necessary column's table is included.",
    ],
    "TableNotFound": [
        "Correct any table name typos or casing issues.",
        "Ensure the referenced table exists in the schema.",
        "Check if table alias is misused or undefined.",
        "Use correct schema.table format if applicable.",
    ],
    "AmbiguousColumn": [
        "Add the table prefix (e.g., table.column) to the ambiguous column.",
        "Remove or rename duplicate columns in SELECT or JOIN.",
        "Ensure aliases don't conflict with real column names.",
    ],
    "NoResult": [
        "Relax WHERE clause filter conditions.",
        "Check for typos in filter values or column names.",
        "Use LIKE or ILIKE for fuzzy matching on strings.",
        "Verify if filters are too strict or exclude valid data.",
        "Check date/time format mismatches.",
    ],
    "TypeMismatch": [
        "Add proper CAST or CONVERT functions between types.",
        "Ensure comparisons use compatible types (e.g., string vs. int).",
        "Check aggregation functions or arithmetic expressions for type issues.",
    ],
    "FunctionNotFound": [
        "Replace undefined functions with valid SQL equivalents.",
        "Check for typos or incorrect number of arguments in functions.",
        "Ensure the dialect supports the used function.",
    ],
    "DuplicateColumn": [
        "Rename columns or use aliases (AS ...) to avoid duplicates.",
        "Ensure SELECT clause does not reference same column multiple times.",
        "Disambiguate columns when using JOINs.",
    ],
    "PermissionDenied": [
        "Check if the user has access to the table or column.",
        "Verify the table or column exists in the schema.",
        "Check if the table or dataset is correct.",
        "If running on shared DB, replace with mock data source.",
        "Simplify the query to avoid restricted columns/tables.",
    ],
    "Timeout": [
        "Add WHERE/LIMIT to restrict rows.",
        "Reduce JOIN complexity or avoid CROSS JOINs.",
        "Replace nested SELECTs with CTEs or flatten logic.",
        "Filter early in subqueries to reduce computation.",
    ],
    "ConstraintError": [
        "Check that JOIN or WHERE clause doesn’t violate foreign key logic.",
        "Ensure uniqueness or NOT NULL constraints are respected.",
        "Avoid violating schema-level integrity rules.",
    ],
    "ResourceExceeded": [
        "Reduce result set size using LIMIT or WHERE.",
        "Simplify aggregations or avoid expensive JOINs.",
        "Avoid SELECT * and target only necessary columns.",
    ],
    "InvalidIdentifier": [
        "Avoid using reserved SQL keywords as column/table names.",
        "Escape problematic names using backticks or quotes if necessary.",
        "Fix special characters or spaces in identifiers.",
    ],
    "NotNullConstraint": [
        "Ensure required columns are filled (no NULL values).",
        "Add COALESCE or default values where appropriate.",
        "Avoid inserting or selecting NULL into NOT NULL fields.",
    ],
    "OtherError": [
        "Try simplifying query structure.",
        "Focus on filtering conditions.",
        "Try SELECT with minimal columns first.",
        "Double-check all referenced schema components.",
    ],
}


def refinement_strategy_selector(error_type: str) -> str:
    """
    Select and format refinement strategies for a given error type.

    Retrieves the appropriate list of refinement strategies based on the error type
    and formats them into a structured prompt for language model guidance.

    Args:
        error_type: The type of SQL error encountered (e.g., "SyntaxError", "ColumnNotFound").

    Returns:
        str: Formatted prompt containing the error type and numbered list of strategies
             to guide the language model in selecting and applying a refinement approach.
    """
    strategies = REFINEMENT_STRATEGIES.get(
        error_type, REFINEMENT_STRATEGIES["OtherError"]
    )
    strategy_prompt = (
        f"[Detected Error Type]: {error_type}\n"
        "Select one of the following strategies to apply:\n"
        + "\n".join([f"{i+1}. {s}" for i, s in enumerate(strategies)])
        + "\n\nExplain your strategy choice and apply it to refine the SQL."
    )
    return strategy_prompt
