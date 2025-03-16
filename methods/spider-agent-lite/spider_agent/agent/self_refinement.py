import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from spider_agent.agent.agents import PromptAgent
from spider_agent.agent.action import Terminate, BIGQUERY_EXEC_SQL, SNOWFLAKE_EXEC_SQL, LOCAL_DB_SQL
from spider_agent.agent.models import call_llm

logger = logging.getLogger("spider_agent")

@dataclass
class RefinementResult:
    """Class to store the result of a refinement iteration"""
    sql_query: str
    result: str
    is_empty: bool = False
    error: bool = False

class SelfRefinementAgent(PromptAgent):
    """
    Extension of PromptAgent with self-refinement capabilities.
    This agent can iteratively refine SQL queries based on execution results
    until termination conditions are met.
    """
    
    def __init__(
        self,
        model="gpt-4",
        max_tokens=1500,
        top_p=0.9,
        temperature=0.5,
        max_memory_length=10,
        max_steps=15,
        use_plan=False,
        max_refinement_iterations=5,  # Maximum number of refinement iterations
    ):
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            max_memory_length=max_memory_length,
            max_steps=max_steps,
            use_plan=use_plan
        )
        
        self.max_refinement_iterations = max_refinement_iterations
        self.refinement_iterations = []
        self.consecutive_empty_results = 0
        self.previous_queries = set()
        
    def run(self):
        """
        Override the run method to include self-refinement logic.
        """
        assert self.env is not None, "Environment is not set."
        result = ""
        done = False
        step_idx = 0
        obs = "You are in the folder now."
        retry_count = 0
        last_action = None
        repeat_action = False
        
        # First run to get initial SQL query
        while not done and step_idx < self.max_steps:
            _, action = self.predict(obs)
            
            if action is None:
                logger.info("Failed to parse action from response, try again.")
                retry_count += 1
                if retry_count > 3:
                    logger.info("Failed to parse action from response, stop.")
                    break
                obs = "Failed to parse action from your response, make sure you provide a valid action."
            else:
                logger.info("Step %d: %s", step_idx + 1, action)
                
                # Check if this is a SQL execution action
                if isinstance(action, (BIGQUERY_EXEC_SQL, SNOWFLAKE_EXEC_SQL, LOCAL_DB_SQL)):
                    # Start self-refinement process
                    refined_action, refinement_obs = self.perform_self_refinement(action)
                    
                    if refined_action is not None:
                        # Use the refined action instead
                        action = refined_action
                        obs = refinement_obs
                        done = isinstance(action, Terminate)
                    else:
                        # If refinement failed, continue with original action
                        obs, done = self.env.step(action)
                elif last_action is not None and last_action == action:
                    if repeat_action:
                        return False, "ERROR: Repeated action"
                    else:
                        obs = "The action is the same as the last one, you MUST provide a DIFFERENT SQL code or Python Code or different action. you MUST provide a DIFFERENT SQL code or Python Code or different action. you MUST provide a DIFFERENT SQL code or Python Code or different action."
                        repeat_action = True
                else:
                    obs, done = self.env.step(action)
                    last_action = action
                    repeat_action = False

            if done:
                if isinstance(action, Terminate):
                    result = action.output
                logger.info("The task is done.")
                break
            step_idx += 1

        return done, result
    
    def perform_self_refinement(self, action) -> Tuple[Optional[Any], str]:
        """
        Perform self-refinement on a SQL query action.
        
        Args:
            action: The SQL action to refine
            
        Returns:
            Tuple of (refined_action, observation)
        """
        logger.info("Starting self-refinement process")
        
        # Extract the SQL query from the action
        if isinstance(action, BIGQUERY_EXEC_SQL):
            sql_query = action.sql_query
            action_type = "BIGQUERY_EXEC_SQL"
            is_save = action.is_save
            save_path = action.save_path
        elif isinstance(action, SNOWFLAKE_EXEC_SQL):
            sql_query = action.sql_query
            action_type = "SNOWFLAKE_EXEC_SQL"
            is_save = action.is_save
            save_path = action.save_path
        elif isinstance(action, LOCAL_DB_SQL):
            sql_query = action.code
            action_type = "LOCAL_DB_SQL"
            file_path = action.file_path
            output = action.output
        else:
            # Not a SQL action, skip refinement
            return None, ""
        
        # Initialize refinement variables
        refinement_iterations = 0
        previous_results = []
        self.consecutive_empty_results = 0
        
        # Add the initial query to previous queries
        self.previous_queries.add(sql_query)
        
        # Execute the initial query
        obs, _ = self.env.step(action)
        
        # Parse the result
        current_result = self._parse_sql_result(obs, sql_query)
        previous_results.append(current_result)
        
        # Check if we need to refine
        if self._should_terminate_refinement(current_result, previous_results):
            logger.info("No refinement needed, terminating refinement process")
            return action, obs
        
        # Start refinement loop
        while refinement_iterations < self.max_refinement_iterations:
            refinement_iterations += 1
            logger.info(f"Refinement iteration {refinement_iterations}")
            
            # Generate refinement prompt
            refinement_prompt = self._generate_refinement_prompt(sql_query, obs, previous_results)
            
            # Call LLM for refinement
            status, refined_sql = self._call_llm_for_refinement(refinement_prompt)
            
            if not status:
                logger.error(f"Failed to call LLM for refinement: {refined_sql}")
                break
            
            # Extract SQL query from the response
            refined_sql = self._extract_sql_from_response(refined_sql)
            
            if not refined_sql or refined_sql in self.previous_queries:
                logger.info("Refined SQL is empty or duplicate, skipping")
                continue
            
            # Add to previous queries
            self.previous_queries.add(refined_sql)
            
            # Create a new action with the refined SQL
            if action_type == "BIGQUERY_EXEC_SQL":
                refined_action = BIGQUERY_EXEC_SQL(sql_query=refined_sql, is_save=is_save, save_path=save_path)
            elif action_type == "SNOWFLAKE_EXEC_SQL":
                refined_action = SNOWFLAKE_EXEC_SQL(sql_query=refined_sql, is_save=is_save, save_path=save_path)
            elif action_type == "LOCAL_DB_SQL":
                refined_action = LOCAL_DB_SQL(code=refined_sql, file_path=file_path, output=output)
            
            # Execute the refined query
            refined_obs, _ = self.env.step(refined_action)
            
            # Parse the result
            current_result = self._parse_sql_result(refined_obs, refined_sql)
            previous_results.append(current_result)
            
            # Store refinement iteration
            self.refinement_iterations.append({
                "iteration": refinement_iterations,
                "sql_query": refined_sql,
                "result": refined_obs,
                "is_empty": current_result.is_empty,
                "error": current_result.error
            })
            
            # Check if we should terminate refinement
            if self._should_terminate_refinement(current_result, previous_results):
                logger.info(f"Terminating refinement after {refinement_iterations} iterations")
                
                # If we have a successful result, create a Terminate action
                if not current_result.error and not current_result.is_empty:
                    if action_type == "BIGQUERY_EXEC_SQL" or action_type == "SNOWFLAKE_EXEC_SQL":
                        if is_save and save_path:
                            return Terminate(output=save_path), f"Self-refinement complete. Final result saved to {save_path}"
                        else:
                            return refined_action, refined_obs
                    else:
                        return refined_action, refined_obs
                else:
                    # Return the best result from previous iterations
                    best_result_idx = self._find_best_result_index(previous_results)
                    if best_result_idx == 0:
                        return action, obs
                    else:
                        best_result = previous_results[best_result_idx]
                        if action_type == "BIGQUERY_EXEC_SQL":
                            best_action = BIGQUERY_EXEC_SQL(sql_query=best_result.sql_query, is_save=is_save, save_path=save_path)
                        elif action_type == "SNOWFLAKE_EXEC_SQL":
                            best_action = SNOWFLAKE_EXEC_SQL(sql_query=best_result.sql_query, is_save=is_save, save_path=save_path)
                        elif action_type == "LOCAL_DB_SQL":
                            best_action = LOCAL_DB_SQL(code=best_result.sql_query, file_path=file_path, output=output)
                        
                        # Re-execute the best action to get its observation
                        best_obs, _ = self.env.step(best_action)
                        return best_action, best_obs
            
            # Update for next iteration
            sql_query = refined_sql
            obs = refined_obs
            action = refined_action
        
        # If we reach max iterations without termination, return the best result
        logger.info(f"Reached maximum refinement iterations ({self.max_refinement_iterations})")
        best_result_idx = self._find_best_result_index(previous_results)
        
        if best_result_idx == 0:
            return action, obs
        else:
            best_result = previous_results[best_result_idx]
            if action_type == "BIGQUERY_EXEC_SQL":
                best_action = BIGQUERY_EXEC_SQL(sql_query=best_result.sql_query, is_save=is_save, save_path=save_path)
            elif action_type == "SNOWFLAKE_EXEC_SQL":
                best_action = SNOWFLAKE_EXEC_SQL(sql_query=best_result.sql_query, is_save=is_save, save_path=save_path)
            elif action_type == "LOCAL_DB_SQL":
                best_action = LOCAL_DB_SQL(code=best_result.sql_query, file_path=file_path, output=output)
            
            # Re-execute the best action to get its observation
            best_obs, _ = self.env.step(best_action)
            return best_action, best_obs
    
    def _parse_sql_result(self, observation: str, sql_query: str) -> RefinementResult:
        """
        Parse the SQL execution result from the observation.
        
        Args:
            observation: The observation from executing the SQL query
            sql_query: The SQL query that was executed
            
        Returns:
            RefinementResult object containing the parsed result
        """
        result = RefinementResult(sql_query=sql_query, result=observation)
        
        # Check for errors
        if "error" in observation.lower() or "exception" in observation.lower():
            result.error = True
            return result
        
        # Check for empty results
        empty_patterns = [
            r"0 rows? affected",
            r"no rows? returned",
            r"empty result",
            r"no results?",
            r"returned 0 rows?"
        ]
        
        for pattern in empty_patterns:
            if re.search(pattern, observation.lower()):
                result.is_empty = True
                self.consecutive_empty_results += 1
                break
        else:
            # Reset consecutive empty results if this result is not empty
            self.consecutive_empty_results = 0
        
        return result
    
    def _should_terminate_refinement(self, current_result: RefinementResult, previous_results: List[RefinementResult]) -> bool:
        """
        Determine if refinement should be terminated based on the termination conditions.
        
        Args:
            current_result: The current refinement result
            previous_results: List of previous refinement results
            
        Returns:
            True if refinement should be terminated, False otherwise
        """
        # Condition 1: Error in the current result
        if current_result.error:
            return False
        
        # Condition 2: Self-consistency - same result twice
        if len(previous_results) >= 2:
            # Check if any previous result matches the current result
            for i in range(len(previous_results) - 1):
                prev_result = previous_results[i]
                if (not prev_result.error and not prev_result.is_empty and 
                    not current_result.error and not current_result.is_empty and
                    self._results_are_equivalent(prev_result.result, current_result.result)):
                    # Only terminate if the result is not 0
                    if "0" not in current_result.result.split("\n")[1]:
                        logger.info("Self-consistency achieved: same result obtained twice")
                        return True
                    else:
                        # Don't terminate if the result is 0, as we know it's incorrect
                        return False
        
        # Condition 3: Consecutive empty results
        if self.consecutive_empty_results >= 2:
            logger.info("Terminating refinement due to consecutive empty results")
            return True
        
        return False
    
    def _results_are_equivalent(self, result1: str, result2: str) -> bool:
        """
        Check if two SQL results are equivalent.
        This is a simplified implementation and might need to be enhanced
        for more complex result comparison.
        
        Args:
            result1: First result string
            result2: Second result string
            
        Returns:
            True if results are equivalent, False otherwise
        """
        # Extract data rows from results (simplified)
        def extract_data(result):
            # This is a simplified extraction - in a real implementation,
            # you would parse the actual data rows from the result
            lines = result.strip().split('\n')
            data_lines = [line for line in lines if '|' in line and not line.startswith('+')]
            return '\n'.join(data_lines)
        
        data1 = extract_data(result1)
        data2 = extract_data(result2)
        
        return data1 == data2
    
    def _generate_refinement_prompt(self, sql_query: str, observation: str, previous_results: List[RefinementResult]) -> str:
        """
        Generate a prompt for the LLM to refine the SQL query.
        
        Args:
            sql_query: The current SQL query
            observation: The observation from executing the SQL query
            previous_results: List of previous refinement results
            
        Returns:
            A prompt string for the LLM
        """
        prompt = f"""You are an expert SQL developer. I need your help to refine the following SQL query based on the execution results.

Task: {self.env.task_config['question']}

Current SQL Query:
```sql
{sql_query}
```

Execution Result:
```
{observation}
```

SQL Refinement Guidelines:
1. Carefully analyze the error messages or empty results
2. Check for syntax errors, incorrect table names, or missing joins
3. Ensure column names are correct and properly referenced
4. Verify that filtering conditions are appropriate for the task
5. Consider using Common Table Expressions (CTEs) to break down complex logic
6. Make sure aggregation functions are used correctly
7. Ensure the output column names match what's expected in the task
8. For date-based queries, verify the date format and filtering approach
9. For BigQuery tables with date suffixes, use _TABLE_SUFFIX for filtering when appropriate

Remember that the goal is to produce a working query that correctly answers the original question.

Previous Refinement Attempts:
"""
        
        # Add previous refinement attempts
        for i, result in enumerate(previous_results[:-1], 1):  # Skip the current result
            prompt += f"""
Attempt {i}:
```sql
{result.sql_query}
```

Result:
```
{result.result}
```
"""
        
        # Add instructions based on the current situation
        if previous_results[-1].error:
            prompt += """
The current query has an error. Please fix the syntax or logical errors in the query.

Common errors to check for:
- Incorrect table or column names
- Missing JOIN conditions
- Syntax errors in functions or expressions
- Incorrect data types in comparisons
- Missing GROUP BY clauses when using aggregation functions
"""
        elif previous_results[-1].is_empty:
            prompt += """
The current query returned empty results. Please modify the query to return meaningful results.

Possible issues to address:
- Filtering conditions might be too restrictive
- JOINs might be eliminating all rows
- Table names or paths might be incorrect
- Date formats or ranges might be incorrect
"""
        else:
            prompt += """
The current query executed successfully but the results may not be correct for the task.

Consider these improvements:
- Check if the logic correctly implements the requirements
- Verify that column selections match what's needed
- Ensure aggregations are calculating the right metrics
- Check if the output format matches what's expected
"""
        
        prompt += """
Please provide a refined SQL query that addresses the issues. Return ONLY the SQL query without any explanations or markdown formatting.
"""
        
        return prompt
    
    def _call_llm_for_refinement(self, prompt: str) -> Tuple[bool, str]:
        """
        Call the LLM to refine the SQL query.
        
        Args:
            prompt: The refinement prompt
            
        Returns:
            Tuple of (success, response)
        """
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an expert SQL developer. Your task is to refine SQL queries based on execution results."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        return call_llm({
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "temperature": self.temperature
        })
    
    def _extract_sql_from_response(self, response: str) -> str:
        """
        Extract the SQL query from the LLM response.
        
        Args:
            response: The LLM response
            
        Returns:
            The extracted SQL query
        """
        # Try to extract SQL from code blocks
        sql_matches = re.findall(r'```(?:sql)?\s*(.*?)\s*```', response, re.DOTALL)
        if sql_matches:
            return sql_matches[0].strip()
        
        # If no code blocks, return the entire response
        return response.strip()
    
    def _find_best_result_index(self, results: List[RefinementResult]) -> int:
        """
        Find the index of the best result from the list of results.
        
        Args:
            results: List of refinement results
            
        Returns:
            Index of the best result
        """
        # Prioritize non-error, non-empty results
        valid_indices = [i for i, result in enumerate(results) if not result.error and not result.is_empty]
        if valid_indices:
            return valid_indices[-1]  # Return the latest valid result
        
        # If no valid results, prioritize non-error results
        non_error_indices = [i for i, result in enumerate(results) if not result.error]
        if non_error_indices:
            return non_error_indices[-1]
        
        # If all have errors, return the first result (original query)
        return 0
    
    def get_trajectory(self):
        """
        Override get_trajectory to include refinement information.
        """
        trajectory = super().get_trajectory()
        
        # Add refinement information
        trajectory["refinement_iterations"] = self.refinement_iterations
        
        return trajectory
