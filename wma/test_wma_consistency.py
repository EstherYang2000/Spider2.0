import json
import os
import logging
from .wma import WeightedMajorityAlgorithm

# Configure logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_test_comparison():
    """
    Run a test comparison between standard WMA and WMA with cross-consistency.
    """
    # Sample data - simulating SQL predictions from different experts
    test_data = [
        {
            "question": "Find all employees who work in the IT department",
            "gold_sql": "SELECT * FROM employees WHERE department = 'IT'",
            "predictions": {
                "expert1": ["SELECT * FROM employees WHERE department = 'IT'", 
                           "SELECT name, id FROM employees WHERE dept = 'IT'"],
                "expert2": ["SELECT * FROM employees WHERE department = 'IT'", 
                           "SELECT id, name, salary FROM employees WHERE department = 'Information Technology'"],
                "expert3": ["SELECT name FROM employees WHERE dept_id = 3", 
                           "SELECT * FROM staff WHERE department = 'IT'"],
                "expert4": ["SELECT id FROM employees WHERE department_name LIKE '%IT%'",
                           "SELECT * FROM workers WHERE dept = 'IT'"]
            }
        },
        {
            "question": "List all products with price greater than $100",
            "gold_sql": "SELECT * FROM products WHERE price > 100",
            "predictions": {
                "expert1": ["SELECT * FROM items WHERE price > 100", 
                           "SELECT name, price FROM products WHERE price > 100"],
                "expert2": ["SELECT * FROM products WHERE price > 100", 
                           "SELECT id, name, price FROM products WHERE price > 100"],
                "expert3": ["SELECT * FROM products WHERE price > 100", 
                           "SELECT * FROM products WHERE cost > 100"],
                "expert4": ["SELECT id FROM inventory WHERE price > 100",
                           "SELECT * FROM products WHERE price >= 100"]
            }
        },
        {
            "question": "Count the number of orders placed in January 2023",
            "gold_sql": "SELECT COUNT(*) FROM orders WHERE order_date BETWEEN '2023-01-01' AND '2023-01-31'",
            "predictions": {
                "expert1": ["SELECT COUNT(*) FROM orders WHERE order_date BETWEEN '2023-01-01' AND '2023-01-31'", 
                           "SELECT COUNT(id) FROM orders WHERE MONTH(order_date) = 1 AND YEAR(order_date) = 2023"],
                "expert2": ["SELECT COUNT(*) FROM sales WHERE date >= '2023-01-01' AND date <= '2023-01-31'", 
                           "SELECT COUNT(order_id) FROM orders WHERE order_date LIKE '2023-01-%'"],
                "expert3": ["SELECT COUNT(*) FROM transactions WHERE transaction_date >= '2023-01-01' AND transaction_date <= '2023-01-31'", 
                           "SELECT COUNT(*) FROM orders WHERE EXTRACT(MONTH FROM order_date) = 1 AND EXTRACT(YEAR FROM order_date) = 2023"],
                "expert4": ["SELECT COUNT(*) FROM orders WHERE order_date BETWEEN '2023-01-01' AND '2023-01-31'",
                           "SELECT COUNT(*) FROM orders WHERE DATE_PART('month', order_date) = 1 AND DATE_PART('year', order_date) = 2023"]
            }
        }
    ]
    
    # Run test with standard WMA (no consistency bonus)
    standard_wma = WeightedMajorityAlgorithm(epsilon=0.01, consistency_bonus=0.0)
    
    # Run test with cross-consistency WMA
    consistency_wma = WeightedMajorityAlgorithm(epsilon=0.01, consistency_bonus=0.05)
    
    # Add experts to both WMAs
    for expert in ["expert1", "expert2", "expert3", "expert4"]:
        standard_wma.add_expert(expert, init_weight=1.0)
        consistency_wma.add_expert(expert, init_weight=1.0)
    
    # Results tracking
    standard_results = []
    consistency_results = []
    
    # Process each test case
    for i, case in enumerate(test_data):
        logger.info(f"Processing test case {i+1}: {case['question']}")
        
        # Get predictions
        predictions = case["predictions"]
        gold_sql = case["gold_sql"]
        
        # Simulate correctness evaluation (in a real scenario, this would use evaluate_cc)
        expert_correctness = {}
        for expert, sql_list in predictions.items():
            # Check if any SQL matches the gold SQL (case-insensitive comparison)
            is_correct = any(sql.lower() == gold_sql.lower() for sql in sql_list)
            expert_correctness[expert] = is_correct
            
            # Update weights in both WMAs
            standard_wma.update_weights(expert, is_correct)
            consistency_wma.update_weights(expert, is_correct)
        
        # Get standard WMA result
        standard_sql, standard_experts, standard_weight = standard_wma.weighted_majority_vote(
            predictions, apply_consistency=False
        )
        
        # Get consistency WMA result
        consistency_sql, consistency_experts, consistency_weight = consistency_wma.weighted_majority_vote(
            predictions, apply_consistency=True
        )
        
        # Check if results are correct
        standard_correct = standard_sql.lower() == gold_sql.lower() if standard_sql else False
        consistency_correct = consistency_sql.lower() == gold_sql.lower() if consistency_sql else False
        
        # Log results
        logger.info(f"Standard WMA result: {standard_correct} (chosen by {standard_experts})")
        logger.info(f"Consistency WMA result: {consistency_correct} (chosen by {consistency_experts})")
        
        # Get consistency scores
        consistency_scores = consistency_wma.calculate_cross_consistency(predictions)
        
        # Store results
        standard_results.append({
            "case": i+1,
            "question": case["question"],
            "chosen_sql": standard_sql,
            "chosen_experts": standard_experts,
            "is_correct": standard_correct,
            "expert_weights": standard_wma.get_weights()
        })
        
        consistency_results.append({
            "case": i+1,
            "question": case["question"],
            "chosen_sql": consistency_sql,
            "chosen_experts": consistency_experts,
            "is_correct": consistency_correct,
            "expert_weights": consistency_wma.get_weights(),
            "consistency_scores": consistency_scores
        })
    
    # Calculate overall accuracy
    standard_accuracy = sum(1 for r in standard_results if r["is_correct"]) / len(standard_results)
    consistency_accuracy = sum(1 for r in consistency_results if r["is_correct"]) / len(consistency_results)
    
    logger.info(f"Standard WMA accuracy: {standard_accuracy:.2f}")
    logger.info(f"Consistency WMA accuracy: {consistency_accuracy:.2f}")
    
    # Print final weights
    logger.info(f"Final standard WMA weights: {standard_wma.get_weights()}")
    logger.info(f"Final consistency WMA weights: {consistency_wma.get_weights()}")
    
    # Save results to file
    os.makedirs("test_results", exist_ok=True)
    with open("test_results/standard_wma_results.json", "w") as f:
        json.dump(standard_results, f, indent=2)
    with open("test_results/consistency_wma_results.json", "w") as f:
        json.dump(consistency_results, f, indent=2)
    
    logger.info("Test results saved to test_results/ directory")
    
    return {
        "standard_accuracy": standard_accuracy,
        "consistency_accuracy": consistency_accuracy,
        "standard_weights": standard_wma.get_weights(),
        "consistency_weights": consistency_wma.get_weights()
    }

if __name__ == "__main__":
    logger.info("Starting WMA cross-consistency test")
    results = run_test_comparison()
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Standard WMA accuracy: {results['standard_accuracy']:.2f}")
    print(f"Consistency WMA accuracy: {results['consistency_accuracy']:.2f}")
    print("\nStandard WMA final weights:")
    for expert, weight in results['standard_weights'].items():
        print(f"  {expert}: {weight:.4f}")
    print("\nConsistency WMA final weights:")
    for expert, weight in results['consistency_weights'].items():
        print(f"  {expert}: {weight:.4f}")
    print("="*50)
