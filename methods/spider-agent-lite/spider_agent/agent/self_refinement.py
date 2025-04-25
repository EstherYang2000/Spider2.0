import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from spider_agent.agent.agents import PromptAgent
from spider_agent.agent.planner_critique_agents import PlannerAgent, CritiqueAgent
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

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

class RefinementLogRAG:
    def __init__(self, log_path="refinement_history.jsonl", model_name="all-MiniLM-L6-v2"):
        self.log_path = log_path
        self.model = SentenceTransformer(model_name)
        self.logs = []
        self.embeddings = None
        self._load_and_embed_logs()

    def _load_and_embed_logs(self):
        import os, json
        self.logs = []
        if not os.path.exists(self.log_path):
            self.embeddings = None
            return
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    case = json.loads(line)
                    self.logs.append(case)
                except Exception:
                    continue
        if self.logs:
            texts = [case.get("original_sql", "") + " " + case.get("refined_sql", "") for case in self.logs]
            self.embeddings = self.model.encode(texts, convert_to_numpy=True)
            self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
            self.index.add(self.embeddings)
        else:
            self.embeddings = None

    def retrieve_similar(self, sql_query, top_k=2):
        if not self.logs or self.embeddings is None:
            return []
        query_emb = self.model.encode([sql_query], convert_to_numpy=True)
        _, indices = self.index.search(query_emb, top_k)
        return [self.logs[idx] for idx in indices[0] if idx < len(self.logs)]


class SelfRefinementAgent(PromptAgent):
    def log_refinement_case(self, original_sql, refined_sql, error, success):
        import json
        record = {
            "original_sql": original_sql,
            "refined_sql": refined_sql,
            "error": error,
            "success": success
        }
        try:
            with open(self.refinement_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log refinement case: {e}")

    def retrieve_similar_refinements(self, sql_query, top_k=2):
        """Retrieve similar past refinements using semantic vector search (RAG)."""
        if not hasattr(self, "_refinement_rag") or self._refinement_rag is None:
            self._refinement_rag = RefinementLogRAG(self.refinement_log_path)
        # Always reload in case log updated (can optimize if needed)
        self._refinement_rag._load_and_embed_logs()
        return self._refinement_rag.retrieve_similar(sql_query, top_k)

    def analyze_sql_error(self, error_msg):
        msg = error_msg.lower() if error_msg else ""
        if "syntax error" in msg or "parse error" in msg:
            return "Syntax error: Please check the SQL syntax."
        elif "no such table" in msg or "table not found" in msg:
            return "Table not found: Please check the table name or database."
        elif ("column" in msg and "not found" in msg) or ("unknown column" in msg):
            return "Column not found: Please check the column names in SELECT/FROM/WHERE clauses."
        elif "permission denied" in msg or "access denied" in msg:
            return "Permission denied: Please check the database access permissions."
        elif "timeout" in msg:
            return "Query timeout: Please optimize the SQL or check the data volume."
        elif "division by zero" in msg:
            return "Division by zero error: Please check the calculation expressions."
        else:
            return "Other error: Please review the message and fix accordingly."

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
        base_top_k=2,                # 新增：RAG檢索案例的預設top_k
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
        self.base_top_k = base_top_k
        self.refinement_iterations = []
        self.consecutive_empty_results = 0
        self.previous_queries = set()
        self.refinement_log_path = "refinement_history.jsonl"  # for self-learning
        self.refinement_rag = RefinementLogRAG(self.refinement_log_path)
        self.planner_agent = PlannerAgent(model=model)
        self.critique_agent = CritiqueAgent(model=model)
        self.plan_critique = None

    def run(self):
        """
        Override the run method to include MCP loop: Planning, Critique, and Multi-step Refinement.
        """
        assert self.env is not None, "Environment is not set."
        result = ""
        done = False
        step_idx = 0
        obs = "You are in the folder now."
        retry_count = 0
        last_action = None
        repeat_action = False
        sql_query = None
        critique_msg = None

        # 1. Always generate a plan at the start
        if not self.reference_plan:
            schema_string = self.env.task_config.get('schema', '')
            evidence = self.env.task_config.get('evidence', '')
            question = self.env.task_config.get('question', '')
            self.reference_plan = self.planner_agent.generate_plan(question, schema_string, evidence)
            logger.info(f"[MCP] Generated Plan: {self.reference_plan}") 
        # Plan critique (only once)
        self.plan_critique = self.critique_agent.critique_sql(
            self.reference_plan,  # 直接用 reference_plan 當作 "sql_query" 參數
            self.reference_plan,  # 也可傳 plan 當作 plan
            self.env.task_config.get('question', ''),
            self.env.task_config.get('schema', ''),
            self.env.task_config.get('evidence', '')
        )
        logger.info(f"[MCP] Plan Critique: {self.plan_critique}")

        # 2. MCP Loop: SQL generation, critique, refinement
        def get_plan_step(plan: str, idx: int) -> str:
            """Extract the idx-th step from a numbered plan string."""
            import re
            steps = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|$)', plan, re.DOTALL)
            if not steps:
                return plan.strip()  # fallback: whole plan if not numbered
            if idx < len(steps):
                return steps[idx].strip()
            else:
                return steps[-1].strip()  # fallback: last step

        while not done and step_idx < self.max_steps:
            # On the first step, include the full reference plan; otherwise, only the current step
            if step_idx == 0:
                prompt = f"Plan:\n{self.reference_plan}"
            else:
                current_plan_step = get_plan_step(self.reference_plan, step_idx)
                prompt = f"Plan (current step):\n{current_plan_step}"
            if self.plan_critique:
                prompt += f"\n\nPlan Critique:\n{self.plan_critique}"
            if critique_msg:
                prompt += f"\n\nPrevious Critique:\n{critique_msg}"
            prompt += f"\n\nObservation:\n{obs}"
            # Use PromptAgent's predict with the structured prompt
            _, action = self.predict(prompt)

            if action is None:
                logger.info("Failed to parse action from response, try again.")
                retry_count += 1
                if retry_count > 3:
                    logger.info("Failed to parse action from response, stop.")
                    break
                obs = "Failed to parse action from your response, make sure you provide a valid action."
                continue

            logger.info("Step %d: %s", step_idx + 1, action)
            # Extract SQL query string if action is SQL execution
            sql_query = getattr(action, 'sql_query', None)

            # 3. Critique the SQL after each generation
            critique_msg = None
            if sql_query:
                schema_string = self.env.task_config.get('schema', '')
                evidence = self.env.task_config.get('evidence', '')
                question = self.env.task_config.get('question', '')
                critique_msg = self.critique_agent.critique_sql(sql_query, self.reference_plan, question, schema_string, evidence)
                logger.info(f"[MCP] Critique: {critique_msg}")

            if last_action is not None and last_action == action:
                if repeat_action:
                    return False, "ERROR: Repeated action"
                else:
                    obs = "The action is the same as the last one, you MUST provide a DIFFERENT SQL code or Python Code or different action."
                    repeat_action = True
            else:
                # === 自動啟用 self-refinement ===
                if hasattr(self, 'self_refinement_enabled') and self.self_refinement_enabled and isinstance(action, (BIGQUERY_EXEC_SQL, SNOWFLAKE_EXEC_SQL, LOCAL_DB_SQL)):
                    action, obs = self.perform_self_refinement(action)
                    # 這裡假設 self.perform_self_refinement 不會直接終止任務（done 由下方邏輯判斷）
                else:
                    obs, done = self.env.step(action)
                last_action = action
                repeat_action = False

            # Optionally: Use critique to refine plan (advanced, not default)
            # (You can add logic here to update self.current_plan based on repeated critique)

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
            refinement_prompt = self._generate_refinement_prompt(sql_query, obs, previous_results, refinement_iterations)
            
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
        Enhanced: Handles consecutive errors, logs error types, and avoids infinite error loops.
        """
        # Track consecutive errors
        if not hasattr(self, 'consecutive_error_count'):
            self.consecutive_error_count = 0
        
        # Condition 1: Error in the current result
        if current_result.error:
            self.consecutive_error_count += 1
            logger.warning(f"[Refinement] Error detected in result (count={self.consecutive_error_count}): {current_result.result}")
            # If too many consecutive errors, terminate refinement
            if self.consecutive_error_count >= 3:
                logger.error("Too many consecutive errors, terminating refinement.")
                return True
            return False
        else:
            self.consecutive_error_count = 0
        
        # Condition 2: Self-consistency - same result twice
        if len(previous_results) >= 2:
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
    
    import pandas as pd
    from io import StringIO
    from difflib import SequenceMatcher

    def _results_are_equivalent(self, result1: str, result2: str) -> bool:
        """
        Compare two SQL results using DataFrame equality, with fallback to string similarity.
        """
        def to_df(result):
            lines = [line for line in result.strip().split('\n') if '|' in line and not line.startswith('+')]
            if not lines:
                return None
            columns = [col.strip() for col in lines[0].split('|') if col.strip()]
            data = []
            for line in lines[1:]:
                row = [cell.strip() for cell in line.split('|') if cell.strip()]
                if row:
                    data.append(row)
            try:
                df = pd.DataFrame(data, columns=columns)
                df = df.sort_values(by=columns).reset_index(drop=True)
                return df
            except Exception:
                return None
        df1 = to_df(result1)
        df2 = to_df(result2)
        if df1 is not None and df2 is not None:
            return df1.equals(df2)
        # fallback: string similarity
        similarity = SequenceMatcher(None, result1.strip(), result2.strip()).ratio()
        return similarity > 0.95

    # Only one _should_terminate_refinement method should exist. Remove duplicate/conflicting definitions.

    def _generate_refinement_prompt(self, sql_query: str, observation: str, previous_results: List[RefinementResult]) -> str:
        """
        Generate a prompt for the LLM to refine the SQL query.
        
        Args:
            sql_query: The current SQL query
            observation: The observation from executing the SQL query
            previous_results: List of previous refinement results
            refinement_iterations: The number of refinement iterations
            
        Returns:
            A prompt string for the LLM
        """
        # 動態調整 top_k：可根據 refinement 次數，或直接用 self.base_top_k
        if refinement_iterations < 2:
            top_k = self.base_top_k
        elif refinement_iterations < 5:
            top_k = max(self.base_top_k, 3)
        else:
            top_k = max(self.base_top_k, 6)
        similar_cases = self.retrieve_similar_refinements(sql_query, top_k=top_k)

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
        9. For BigQuery tables with date suffixes, use _TABLE_SUFFIX for filtering when appropriate"""
        if similar_cases:
            prompt += "\nRelevant Past Refinements:\n"
            for case in similar_cases:
                prompt += f"- Original SQL: {case.get('original_sql','')[:80]}...\n"
                prompt += f"  Refined SQL: {case.get('refined_sql','')[:80]}...\n"
                prompt += f"  Error: {case.get('error','')}\n"
        if previous_results and len(previous_results) > 1:
            prompt += "\nPrevious Refinement Attempts:\n"
            for idx, res in enumerate(previous_results[:-1]):
                prompt += f"- Iteration {idx+1}: SQL: {res.sql_query[:120]}... | Result: {res.result[:120]}...\n"
        else:
            prompt += "\nNo previous refinement results.\n"
        
        prompt += "\nPlease provide a refined SQL query that addresses the issues. Return ONLY the SQL query without any explanations or markdown formatting.\n"
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
