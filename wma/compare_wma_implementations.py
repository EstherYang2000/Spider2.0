import logging
import json
import os
from .wma_original import WeightedMajorityAlgorithm as OriginalWMA
from .wma_cross_consistency import WeightedMajorityAlgorithm as CrossConsistencyWMA

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compare_implementations():
    """
    Compare the original WMA implementation with the cross-consistency enhanced version.
    """
    # Test data with various scenarios to highlight differences
    test_data = [
        # Case 1: Simple case where experts mostly agree
        {
            "question": "Find all employees in the IT department",
            "gold_sql": "SELECT * FROM employees WHERE department = 'IT'",
            "predictions": {
                "expert1": ["SELECT * FROM employees WHERE department = 'IT'"],
                "expert2": ["SELECT * FROM employees WHERE department = 'IT'"],
                "expert3": ["SELECT name FROM employees WHERE dept_id = 3"],
                "expert4": ["SELECT * FROM employees WHERE department = 'IT'"]
            }
        },
        # Case 2: Experts are split
        {
            "question": "List all products with price greater than $100",
            "gold_sql": "SELECT * FROM products WHERE price > 100",
            "predictions": {
                "expert1": ["SELECT * FROM products WHERE price > 100"],
                "expert2": ["SELECT * FROM items WHERE price > 100"],
                "expert3": ["SELECT * FROM products WHERE price > 100"],
                "expert4": ["SELECT * FROM items WHERE price > 100"]
            }
        },
        # Case 3: Majority is wrong, minority is right
        {
            "question": "Find the total revenue for each product category",
            "gold_sql": "SELECT category, SUM(price * quantity) FROM sales JOIN products ON sales.product_id = products.id GROUP BY category",
            "predictions": {
                "expert1": ["SELECT category, SUM(price * quantity) FROM sales JOIN products ON sales.product_id = products.id GROUP BY category"],
                "expert2": ["SELECT category, SUM(revenue) FROM sales GROUP BY category"],
                "expert3": ["SELECT category, SUM(revenue) FROM sales GROUP BY category"],
                "expert4": ["SELECT category, SUM(revenue) FROM sales GROUP BY category"]
            }
        },
        # Case 4: One expert is consistently right, others are consistently wrong
        {
            "question": "Count the number of orders placed in January 2023",
            "gold_sql": "SELECT COUNT(*) FROM orders WHERE order_date BETWEEN '2023-01-01' AND '2023-01-31'",
            "predictions": {
                "expert1": ["SELECT COUNT(*) FROM orders WHERE order_date BETWEEN '2023-01-01' AND '2023-01-31'"],
                "expert2": ["SELECT COUNT(*) FROM sales WHERE date >= '2023-01-01' AND date <= '2023-01-31'"],
                "expert3": ["SELECT COUNT(*) FROM transactions WHERE transaction_date >= '2023-01-01' AND transaction_date <= '2023-01-31'"],
                "expert4": ["SELECT COUNT(*) FROM invoices WHERE invoice_date LIKE '2023-01-%'"]
            }
        },
        # Case 5: Experts with varying degrees of consistency
        {
            "question": "Find the average salary of employees in each department",
            "gold_sql": "SELECT department, AVG(salary) FROM employees GROUP BY department",
            "predictions": {
                "expert1": ["SELECT department, AVG(salary) FROM employees GROUP BY department"],
                "expert2": ["SELECT department, AVG(salary) FROM employees GROUP BY department"],
                "expert3": ["SELECT dept, AVG(salary) FROM staff GROUP BY dept"],
                "expert4": ["SELECT department, AVERAGE(salary) FROM employees GROUP BY department"]
            }
        }
    ]
    
    # Initialize both implementations
    original_wma = OriginalWMA(epsilon=0.01)
    cross_consistency_wma = CrossConsistencyWMA(epsilon=0.01, consistency_bonus=0.05)
    
    # Add the same experts to both
    for expert in ["expert1", "expert2", "expert3", "expert4"]:
        original_wma.add_expert(expert, init_weight=1.0)
        cross_consistency_wma.add_expert(expert, init_weight=1.0)
    
    # Results tracking
    original_results = []
    cross_consistency_results = []
    
    # Process each test case
    for i, case in enumerate(test_data):
        logger.info(f"Processing test case {i+1}: {case['question']}")
        
        # Get predictions and gold SQL
        predictions = case["predictions"]
        gold_sql = case["gold_sql"]
        
        # Get original WMA result
        original_sql, original_experts, original_weight = original_wma.weighted_majority_vote(predictions)
        
        # Get cross-consistency WMA result
        cross_consistency_sql, cross_consistency_experts, cross_consistency_weight = cross_consistency_wma.weighted_majority_vote(
            predictions, apply_consistency=True
        )
        
        # Check if results are correct
        original_correct = original_sql.lower() == gold_sql.lower() if original_sql else False
        cross_consistency_correct = cross_consistency_sql.lower() == gold_sql.lower() if cross_consistency_sql else False
        
        # Calculate consistency scores for reporting
        consistency_scores = cross_consistency_wma.calculate_cross_consistency(predictions)
        
        # Simulate correctness evaluation for each expert
        expert_correctness = {}
        for expert, sql_list in predictions.items():
            # Check if any SQL matches the gold SQL (case-insensitive comparison)
            is_correct = any(sql.lower() == gold_sql.lower() for sql in sql_list)
            expert_correctness[expert] = is_correct
            
            # Update weights in both implementations
            original_wma.update_weights(expert, is_correct)
            cross_consistency_wma.update_weights(expert, is_correct)
        
        # Log results
        logger.info(f"Case {i+1} original WMA result: {original_correct} (chosen by {original_experts})")
        logger.info(f"Case {i+1} cross-consistency WMA result: {cross_consistency_correct} (chosen by {cross_consistency_experts})")
        
        # Store results
        original_results.append({
            "case": i+1,
            "question": case["question"],
            "chosen_sql": original_sql,
            "chosen_experts": original_experts,
            "is_correct": original_correct,
            "expert_correctness": expert_correctness,
            "expert_weights": original_wma.get_weights()
        })
        
        cross_consistency_results.append({
            "case": i+1,
            "question": case["question"],
            "chosen_sql": cross_consistency_sql,
            "chosen_experts": cross_consistency_experts,
            "is_correct": cross_consistency_correct,
            "expert_correctness": expert_correctness,
            "expert_weights": cross_consistency_wma.get_weights(),
            "consistency_scores": consistency_scores
        })
    
    # Calculate overall accuracy
    original_accuracy = sum(1 for r in original_results if r["is_correct"]) / len(original_results)
    cross_consistency_accuracy = sum(1 for r in cross_consistency_results if r["is_correct"]) / len(cross_consistency_results)
    
    logger.info(f"Original WMA accuracy: {original_accuracy:.2f}")
    logger.info(f"Cross-consistency WMA accuracy: {cross_consistency_accuracy:.2f}")
    
    # Save results to file
    os.makedirs("comparison_results", exist_ok=True)
    with open("comparison_results/original_wma_results.json", "w") as f:
        json.dump(original_results, f, indent=2)
    with open("comparison_results/cross_consistency_wma_results.json", "w") as f:
        json.dump(cross_consistency_results, f, indent=2)
    
    # Generate comparison summary
    comparison_summary = {
        "test_cases": len(test_data),
        "original_wma": {
            "accuracy": original_accuracy,
            "final_weights": original_wma.get_weights()
        },
        "cross_consistency_wma": {
            "accuracy": cross_consistency_accuracy,
            "final_weights": cross_consistency_wma.get_weights(),
            "consistency_scores": cross_consistency_wma.get_consistency_scores()
        },
        "case_by_case_comparison": []
    }
    
    # Add case-by-case comparison
    for i in range(len(test_data)):
        original_result = original_results[i]
        cross_consistency_result = cross_consistency_results[i]
        
        comparison_summary["case_by_case_comparison"].append({
            "case": i+1,
            "question": original_result["question"],
            "original_wma_correct": original_result["is_correct"],
            "cross_consistency_wma_correct": cross_consistency_result["is_correct"],
            "original_wma_experts": original_result["chosen_experts"],
            "cross_consistency_wma_experts": cross_consistency_result["chosen_experts"],
            "expert_correctness": original_result["expert_correctness"],
            "consistency_scores": cross_consistency_result.get("consistency_scores", {})
        })
    
    with open("comparison_results/comparison_summary.json", "w") as f:
        json.dump(comparison_summary, f, indent=2)
    
    logger.info("Comparison results saved to comparison_results/ directory")
    
    return comparison_summary

def print_comparison_table(summary):
    """
    Print a formatted comparison table of the two implementations.
    """
    print("\n" + "="*80)
    print("COMPARISON OF WMA IMPLEMENTATIONS")
    print("="*80)
    
    print(f"\nTest Cases: {summary['test_cases']}")
    print(f"Original WMA Accuracy: {summary['original_wma']['accuracy']:.2f}")
    print(f"Cross-Consistency WMA Accuracy: {summary['cross_consistency_wma']['accuracy']:.2f}")
    
    print("\nCase-by-Case Comparison:")
    print("-" * 80)
    print(f"{'Case':^5} | {'Question':^30} | {'Original':^10} | {'Cross-Consistency':^20} | {'Difference':^10}")
    print("-" * 80)
    
    for case in summary["case_by_case_comparison"]:
        question = case["question"]
        if len(question) > 27:
            question = question[:24] + "..."
            
        original_correct = "✓" if case["original_wma_correct"] else "✗"
        cross_correct = "✓" if case["cross_consistency_wma_correct"] else "✗"
        
        difference = ""
        if case["original_wma_correct"] != case["cross_consistency_wma_correct"]:
            difference = "Cross-Con Win" if case["cross_consistency_wma_correct"] else "Original Win"
        
        print(f"{case['case']:^5} | {question:^30} | {original_correct:^10} | {cross_correct:^20} | {difference:^10}")
    
    print("-" * 80)
    
    print("\nFinal Expert Weights:")
    print("-" * 80)
    print(f"{'Expert':^10} | {'Original WMA':^15} | {'Cross-Consistency WMA':^25} | {'Consistency Score':^20}")
    print("-" * 80)
    
    for expert in summary["original_wma"]["final_weights"]:
        original_weight = summary["original_wma"]["final_weights"][expert]
        cross_weight = summary["cross_consistency_wma"]["final_weights"][expert]
        consistency_score = summary["cross_consistency_wma"]["consistency_scores"].get(expert, 0.0)
        
        print(f"{expert:^10} | {original_weight:.4f}{'':^8} | {cross_weight:.4f}{'':^18} | {consistency_score:.4f}{'':^13}")
    
    print("-" * 80)
    
    print("\nKey Differences:")
    print("1. Cross-Consistency WMA rewards experts that agree with others")
    print("2. Cross-Consistency WMA can potentially overcome incorrect majority votes")
    print("3. Cross-Consistency WMA maintains a history of expert agreement patterns")
    print("4. Cross-Consistency WMA allows tuning the balance between correctness and consistency")
    
    print("\nRecommended Use Cases:")
    print("- Original WMA: When expert correctness is the only important factor")
    print("- Cross-Consistency WMA: When expert agreement patterns provide valuable signals")
    print("  especially in domains where consensus among experts often indicates correctness")
    
    print("="*80)

if __name__ == "__main__":
    logger.info("Starting comparison of WMA implementations")
    summary = compare_implementations()
    print_comparison_table(summary)
