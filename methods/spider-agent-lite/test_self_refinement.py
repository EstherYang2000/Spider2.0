#!/usr/bin/env python3
"""
Test script for the self-refinement feature in Spider-agent-lite.
This script runs a single example with self-refinement enabled.

Usage:
    python test_self_refinement.py --model gpt-4o --example_index 0
"""

import argparse
import os
import sys
import logging

from run import test, config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    # Get default arguments from config
    args = config()
    
    # Override with command line arguments
    parser = argparse.ArgumentParser(
        description="Test the self-refinement feature with a single example"
    )
    
    parser.add_argument("--model", type=str, default="gpt-4o", 
                        help="Model to use (default: gpt-4o)")
    parser.add_argument("--example_index", "-i", type=str, default="0", 
                        help="Index of the example to run (default: 0)")
    parser.add_argument("--max_refinement_iterations", type=int, default=3, 
                        help="Maximum number of refinement iterations (default: 3)")
    parser.add_argument("--test_path", "-t", type=str, default="./examples/spider2-lite.jsonl",
                        help="Path to the test examples (default: ./examples/spider2-lite.jsonl)")
    parser.add_argument("--suffix", "-s", type=str, default="test-refinement",
                        help="Suffix for the experiment ID (default: test-refinement)")
    
    cmd_args = parser.parse_args()
    
    # Update args with command line arguments
    args.model = cmd_args.model
    args.example_index = cmd_args.example_index
    args.max_refinement_iterations = cmd_args.max_refinement_iterations
    args.test_path = cmd_args.test_path
    args.suffix = cmd_args.suffix
    
    # Enable self-refinement
    args.self_refinement = True
    
    # Enable overwriting of existing results
    args.overwriting = True
    
    # Run the test
    logger.info(f"Running test with model {args.model} and example index {args.example_index}")
    logger.info(f"Self-refinement enabled with max iterations: {args.max_refinement_iterations}")
    
    test(args)
    
    logger.info("Test completed")

if __name__ == "__main__":
    main()
