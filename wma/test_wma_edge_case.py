import json
import os
import logging
from .wma import WeightedMajorityAlgorithm

# Configure logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_edge_case_test():
    """
    Run a test with edge cases where cross-consistency can make a difference.
    """
    # Edge case test data designed to show differences
    test_data = [
        # Training case 1: Expert1 and Expert4 are correct
        {
            "question": "Find all employees in the IT department",
            "gold_sql": "SELECT * FROM employees WHERE department = 'IT'",
            "predictions": {
                "expert1": ["SELECT * FROM employees WHERE department = 'IT'"],
                "expert2": ["SELECT * FROM staff WHERE dept = 'IT'"],
                "expert3": ["SELECT * FROM staff WHERE dept = 'IT'"],
                "expert4": ["SELECT * FROM employees WHERE department = 'IT'"]
            }
        },
        # Training case 2: Expert1 and Expert4 are correct again
        {
            "question": "List all products with price greater than $100",
            "gold_sql": "SELECT * FROM products WHERE price > 100",
            "predictions": {
                "expert1": ["SELECT * FROM products WHERE price > 100"],
                "expert2": ["SELECT * FROM items WHERE price > 100"],
                "expert3": ["SELECT * FROM items WHERE price > 100"],
                "expert4": ["SELECT * FROM products WHERE price > 100"]
            }
        },
        # Critical case: Expert2 and Expert3 agree but are wrong, Expert1 and Expert4 disagree but one is right
        # This is where consistency can hurt if not balanced with accuracy
        {
            "question": "Find the total revenue for each product category",
            "gold_sql": "SELECT category, SUM(price * quantity) FROM sales JOIN products ON sales.product_id = products.id GROUP BY category",
            "predictions": {
                "expert1": ["SELECT category, SUM(price * quantity) FROM sales JOIN products ON sales.product_id = products.id GROUP BY category"],
                "expert2": ["SELECT category, SUM(revenue) FROM sales GROUP BY category"],
                "expert3": ["SELECT category, SUM(revenue) FROM sales GROUP BY category"],
                "expert4": ["SELECT p.category_name, SUM(s.price * s.quantity) FROM sales s JOIN products p ON s.product_id = p.id GROUP BY p.category_name"]
            }
        }
    ]
    
    # Run multiple trials with different consistency bonus values
    consistency_values = [0.0, 0.01, 0.03, 0.05, 0.1]
    results = {}
    
    for consistency_bonus in consistency_values:
        # Initialize WMA with current consistency bonus
        wma = WeightedMajorityAlgorithm(epsilon=0.01, consistency_bonus=consistency_bonus)
        
        # Add experts
        for expert in ["expert1", "expert2", "expert3", "expert4"]:
            wma.add_expert(expert, init_weight=1.0)
        
        # Process each test case
        case_results = []
        for i, case in enumerate(test_data):
            logger.info(f"Processing test case {i+1} with consistency_bonus={consistency_bonus}")
            
            # Get predictions
            predictions = case["predictions"]
            gold_sql = case["gold_sql"]
            
            # Calculate consistency scores before voting
            consistency_scores = wma.calculate_cross_consistency(predictions)
            
            # Get WMA result
            final_sql, chosen_experts, best_weight = wma.weighted_majority_vote(
                predictions, apply_consistency=(consistency_bonus > 0)
            )
            
            # Check if result is correct
            is_correct = final_sql.lower() == gold_sql.lower() if final_sql else False
            
            # Simulate correctness evaluation for each expert
            expert_correctness = {}
            for expert, sql_list in predictions.items():
                # Check if any SQL matches the gold SQL (case-insensitive comparison)
                is_expert_correct = any(sql.lower() == gold_sql.lower() for sql in sql_list)
                expert_correctness[expert] = is_expert_correct
                
                # Update weights
                wma.update_weights(expert, is_expert_correct)
            
            # Log results
            logger.info(f"Case {i+1} result with consistency_bonus={consistency_bonus}: {is_correct} (chosen by {chosen_experts})")
            logger.info(f"Current weights: {wma.get_weights()}")
            
            # Store results
            case_results.append({
                "case": i+1,
                "question": case["question"],
                "chosen_sql": final_sql,
                "chosen_experts": chosen_experts,
                "is_correct": is_correct,
                "expert_correctness": expert_correctness,
                "expert_weights": wma.get_weights(),
                "consistency_scores": consistency_scores
            })
        
        # Calculate accuracy
        accuracy = sum(1 for r in case_results if r["is_correct"]) / len(case_results)
        logger.info(f"Accuracy with consistency_bonus={consistency_bonus}: {accuracy:.2f}")
        
        # Store results for this consistency bonus value
        results[consistency_bonus] = {
            "case_results": case_results,
            "accuracy": accuracy,
            "final_weights": wma.get_weights()
        }
    
    # Save results to file
    os.makedirs("test_results", exist_ok=True)
    with open("test_results/edge_case_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Edge case test results saved to test_results/edge_case_results.json")
    
    return results

if __name__ == "__main__":
    logger.info("Starting WMA cross-consistency edge case test")
    results = run_edge_case_test()
    
    # Print summary
    print("\n" + "="*50)
    print("EDGE CASE TEST SUMMARY")
    print("="*50)
    print("Accuracy with different consistency bonus values:")
    for bonus, data in results.items():
        print(f"  consistency_bonus={bonus}: {data['accuracy']:.2f}")
    
    print("\nFinal weights with different consistency bonus values:")
    for bonus, data in results.items():
        print(f"\nconsistency_bonus={bonus}:")
        for expert, weight in data['final_weights'].items():
            print(f"  {expert}: {weight:.4f}")
    print("="*50)
