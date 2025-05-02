import base64
import json
import logging
import os
import re
import time
import uuid
from http import HTTPStatus
from io import BytesIO
from typing import Dict, List
from spider_agent.agent.prompts import BIGQUERY_SYSTEM, LOCAL_SYSTEM, DBT_SYSTEM, SNOWFLAKE_SYSTEM, REFERENCE_PLAN_SYSTEM,EXTERNAL_KNOWLEDGE_SYSTEM
from spider_agent.agent.action import Action, Bash, Terminate, CreateFile, EditFile, LOCAL_DB_SQL, BIGQUERY_EXEC_SQL, SNOWFLAKE_EXEC_SQL, BQ_GET_TABLES, BQ_GET_TABLE_INFO, BQ_SAMPLE_ROWS
from spider_agent.envs.spider_agent import Spider_Agent_Env
from spider_agent.agent.models import call_llm


from openai import AzureOpenAI
from typing import Dict, List, Optional, Tuple, Any, TypedDict

from spider_agent.agent.rag_action import RAG_QUERY
from spider_agent.agent.schema_link_agent import SchemaLinkAgent  # <-- 新增
from spider_agent.agent.schema_agent import SchemaAgent, SchemaAgentEnv


logger = logging.getLogger("spider_agent")

def _infer_dialect_from_env(env) -> str:
    """推斷 SQL dialect，根據 instance_id 前綴或其他 task_config 設定。"""
    db_type = env.task_config.get('instance_id', '').lower()

    if db_type.startswith("local") or "sqlite" in db_type:
        return "sqlite"
    if db_type.startswith("sf") or "snowflake" in db_type:
        return "snowflake"
    if db_type.startswith("bq") or db_type.startswith("ga") or "bigquery" in db_type:
        return "bigquery"
    if "pg" in db_type or "postgres" in db_type:
        return "postgres"
    if "mysql" in db_type:
        return "mysql"
    
    return "generic"  # 預設 fallback

def critique_needs_schema(critique_notes):
    schema_keywords = [
        "missing column", "missing table", "unknown column", "unknown table",
        "schema insufficient", "no such column", "no such table",
        "not found in schema", "schema missing", "column not found", "table not found"
    ]
    for note in critique_notes:
        note_lower = note.lower()
        if any(kw in note_lower for kw in schema_keywords):
            return True
    return False

class PromptAgent:
    def __init__(
        self,
        model="gpt-4",
        max_tokens=1500,
        top_p=0.9,
        temperature=0.5,
        max_memory_length=10,
        max_steps=15,
        use_plan=False,
        use_schema_linking=False,
        env=None,
        llm_predict=None
    ):
        
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.max_memory_length = max_memory_length
        self.max_steps = max_steps
        
        self.thoughts = []
        self.responses = []
        self.actions = []
        self.observations = []
        self.system_message = ""
        self.history_messages = []
        self.env = env
        self.codes = []
        self.work_dir = "/workspace"
        self.reference_plan = None
        self.use_plan = use_plan
        self.use_schema_linking = use_schema_linking
        self.schema_retriever = None  # 兩階段 schema 檢索器（延遲初始化）
        self.dialect = None
        self.schema_string = None
        # --- 初始化 schema_agent ---
        self.schema_agent_env = None
        if llm_predict is None:
            def llm_predict(obs):
                raise NotImplementedError("llm_predict function must be provided!")
        self.schema_agent = None
    
    def find_ddl_csv(self, example_dir, db):
        for root, dirs, files in os.walk(example_dir):
            for file in files:
                if file.lower() == "ddl.csv" and db in root:
                    return (os.path.join(root, file), root)
        return None
    def generate_reference_plan(self):
        from spider_agent.agent.planner_critique_agents import PlannerAgent
        if self.planner_agent is None:
            self.planner_agent = PlannerAgent(model=self.model)
        if not self.reference_plan:
            question = self.env.task_config.get('question', '')
            schema_string = self.env.task_config.get('schema', '')
            evidence = self.env.task_config.get('evidence', '')
            self.reference_plan = self.planner_agent.generate_plan(
                question=question,
                schema_string=schema_string,
                evidence=evidence
            )
        logger.info(f"[MCP] Generated Plan: {self.reference_plan}")   
    def set_env_and_task(self, env: Spider_Agent_Env):
        self.env = env
        self.thoughts = []
        self.responses = []
        self.actions = []
        self.observations = []
        self.codes = []
        self.history_messages = []
        self.instruction = self.env.task_config['question']

        external_knowledge_content = None
        # Check if external knowledge is available
        if 'external_knowledge' in self.env.task_config and self.env.task_config['external_knowledge'] is not None:
            knowledge_file = self.env.task_config['external_knowledge']
            knowledge_path = os.path.join("../../spider2-lite/resource/documents", knowledge_file)

            if os.path.exists(knowledge_path):
                # 🔹 Initialize RAG_QUERY instance with the exact document
                rag_query_action = RAG_QUERY(query=self.instruction, top_k=9)

                # 🔹 Retrieve relevant knowledge **only from the specific file**
                external_knowledge_content = rag_query_action.retrieve_relevant_knowledge(knowledge_path)
                self.env.task_config['evidence'] = external_knowledge_content
                # 🔹 Add the RAG_QUERY action to the agent's memory
                # self.actions.append(rag_query_action)
        
  
        self.dialect = _infer_dialect_from_env(self.env)

            
        if self.env.task_config['type'] == 'Bigquery':
            self._AVAILABLE_ACTION_CLASSES = [Bash, Terminate, BIGQUERY_EXEC_SQL, CreateFile, EditFile]
            action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
            self.system_message = BIGQUERY_SYSTEM.format(work_dir=self.work_dir, action_space=action_space, task=self.instruction, max_steps=self.max_steps)
        elif self.env.task_config['type'] == 'Snowflake':
            self._AVAILABLE_ACTION_CLASSES = [Bash, Terminate, SNOWFLAKE_EXEC_SQL, CreateFile, EditFile]
            action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
            self.system_message = SNOWFLAKE_SYSTEM.format(work_dir=self.work_dir, action_space=action_space, task=self.instruction, max_steps=self.max_steps)
        elif self.env.task_config['type'] == 'Local':
            self._AVAILABLE_ACTION_CLASSES = [Bash, Terminate, CreateFile, EditFile, LOCAL_DB_SQL]
            action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
            self.system_message = LOCAL_SYSTEM.format(work_dir=self.work_dir, action_space=action_space, task=self.instruction, max_steps=self.max_steps)
        elif self.env.task_config['type'] == 'DBT':
            self._AVAILABLE_ACTION_CLASSES = [Bash, Terminate, CreateFile, EditFile, LOCAL_DB_SQL]
            action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
            self.system_message = DBT_SYSTEM.format(work_dir=self.work_dir, action_space=action_space, task=self.instruction, max_steps=self.max_steps)
        
                # # ==== Schema Linking 聚焦 ====
        import pandas as pd
        if getattr(self, "use_schema_linking", False):
            # try:
            #     ddl_path = None
            #     data_path = None
            #     result = self.find_ddl_csv(self.env.mnt_dir, self.env.task_config.get('db'))
            #     logger.info(f"[MCP] DDL Path: {result}")
            #     if result:
            #         ddl_path = result[0]
            #         data_path = result[1]
            #     if ddl_path and data_path:
            #         if self.schema_retriever is None:
            #             self.schema_retriever = SchemaLinkAgent()
            #         schema = pd.read_csv(ddl_path)
            #         logger.info(f"[MCP] Schema: {schema.head(1)}")
            #         logger.info(f"[MCP] Data Path: {data_path}")
            #         logger.info(f"[MCP] Instruction: {self.instruction}")
            #         schema_string = self.schema_retriever.link(
            #             self.instruction,
            #             schema,
            #             data_path
            #         )
            #         self.env.task_config['schema'] = schema_string
            #         self.schema_string = schema_string
            #         logger.info(f"[MCP] Schema String: {schema_string}")
            #     else:
            #         logger.warning("No DDL/schema path found, skipping two-stage schema retrieval.")

            # except Exception as e:
            #     logger.warning(f"[Auto Schema] Failed to generate schema string: {e}")
            # 自動補 schema（如果還沒 schema）
                    # --- 初始化 schema_agent ---
            self.schema_agent_env = SchemaAgentEnv(base_dir=self.env.mnt_dir)
            self.schema_agent = SchemaAgent(self.schema_agent_env, llm_predict=self.predict)
            if not self.schema_string:
                logger.info("[MCP] No schema found, invoking SchemaAgent to generate schema...")
                
                self.schema_string = self.schema_agent.run(
                    user_question=self.env.task_config.get('question', self.instruction),
                    critique_note=None  # 第一次通常沒有 critique
                )
                # self.schema_string = self.schema_agent.format_schema_prompt(self.schema_string)

                logger.info(f"SchemaAgent generated new schema: {self.schema_string}")
                self.env.task_config['schema'] = self.schema_string

        
        logger.info("Reference_plan: %s", self.use_plan)


        # --- Planning and Critique Integration ---
        if self.use_plan:
            # If no plan is provided, generate one using PlannerAgent and critique using CritiqueAgent
            try:
                logger.info("Generating plan...")
                # Generate plan (with or without schema linking)
                self.generate_reference_plan()
                logger.info("Generated plan in prompt agent: %s", self.reference_plan)
                self.system_message += REFERENCE_PLAN_SYSTEM.format(plan=self.reference_plan)

            except Exception as e:
                import traceback
                self.system_message += f"\n[ERROR: Failed to auto-generate plan/critique: {e}\n{traceback.format_exc()}]"

        if external_knowledge_content is not None:
            self.system_message += EXTERNAL_KNOWLEDGE_SYSTEM.format(knowledge=external_knowledge_content)
        
        self.history_messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": self.system_message 
                },
            ]
        })
        
    def predict(self, obs: Dict=None) -> List:
        """
        Predict the next action(s) based on the current observation.
        """    
        
        assert len(self.observations) == len(self.actions) and len(self.actions) == len(self.thoughts) \
            , "The number of observations and actions should be the same."

        status = False
        while not status:
            messages = self.history_messages.copy()
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Observation: {}\n".format(str(obs))
                    }
                ]
            })  
            status, response = call_llm({
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "temperature": self.temperature
            })
            response = response.strip()
            if not status:
                if response in ["context_length_exceeded","rate_limit_exceeded","max_tokens","unknown_error"]:
                    self.history_messages = [self.history_messages[0]] + self.history_messages[3:]
                else:
                    raise Exception(f"Failed to call LLM, response: {response}")
            

        try:
            action = self.parse_action(response)
            thought = re.search(r'Thought:(.*?)Action', response, flags=re.DOTALL)
            if thought:
                thought = thought.group(1).strip()
            else:
                thought = response
        except ValueError as e:
            print("Failed to parse action from response", e)
            action = None
        logger.info("Thought: %s", thought)
        logger.info("Observation: %s", obs)
        logger.info("Response: %s", response)

        self._add_message(obs, thought, action)
        self.observations.append(obs)
        self.thoughts.append(thought)
        self.responses.append(response)
        self.actions.append(action)

        # if action is not None:
        #     self.codes.append(action.code)
        # else:
        #     self.codes.append(None)

        return response, action
        
    
    def _add_message(self, observations: str, thought: str, action: Action):
        self.history_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Observation: {}".format(observations)
                }
            ]
        })
        self.history_messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Thought: {}\n\nAction: {}".format(thought, str(action))
                }
            ]
        })
        if len(self.history_messages) > self.max_memory_length*2+1:
            self.history_messages = [self.history_messages[0]] + self.history_messages[-self.max_memory_length*2:]
    
    def parse_action(self, output) -> Action:
        """ Parse action from text or dict (robust for LLM output) """
        import re

        action_string = None
        # Multi-line robust action extraction
        multiline_patterns = [
            r'Action\s*:\s*((?:.|\n)*?)(?=^Thought:|^Observation:|\Z)',  # up to next block or end
            r'Action\s*:\s*((?:.|\n)*)' , # fallback: grab everything after Action:
            r'Action\s*:\s*((?:.|\n)*?)(?=^Thought:|^Observation:|^\[\d{4}-|\Z)',
            r'Action\s*:\s*([A-Z_]+\s*\(.*)',
            r'Action\s*:\s*((?:[A-Z_]+\s*\(.*?^\))|(?:[A-Z_]+\s*\(.*\Z))',
            r'Action\s*:\s*([A-Z_]+\s*\((?:.|\n)*?\))'


        ]
        

        # If output is a dict (e.g., {'thought':..., 'action':..., 'response':...})
        if isinstance(output, dict):
            action_string = output.get('action')
            # If action string is None or empty, try to extract from response
            if not action_string or action_string == "None":
                resp = output.get('response', '')
                for p in multiline_patterns:
                    match = re.search(p, resp, flags=re.DOTALL | re.MULTILINE)
                    if match:
                        action_string = match.group(1).strip()
                        break
                if not action_string:
                    action_string = resp.strip()
        else:
            # output is a string
            action_string = ""
            for p in multiline_patterns:
                match = re.search(p, output, flags=re.DOTALL | re.MULTILINE)
                if match:
                    action_string = match.group(1).strip()
                    break
            if not action_string:
                action_string = output.strip()
            # logger.info("Parsed action string: %s", action_string)
        output_action = None
        for action_cls in self._AVAILABLE_ACTION_CLASSES:
            action = action_cls.parse_action_from_text(action_string)
            if action is not None:
                output_action = action
                break
        # logger.info("Parsed action: %s", output_action)
        if output_action is None:
            action_string = action_string.replace("\_", "_").replace("'''","```")
            for action_cls in self._AVAILABLE_ACTION_CLASSES:
                action = action_cls.parse_action_from_text(action_string)
                if action is not None:
                    output_action = action
                    break
        logger.info("Parsed action: %s", output_action)
        return output_action

    

    
    def run(self):
        assert self.env is not None, "Environment is not set."
        result = ""
        done = False
        step_idx = 0
        obs = "You are in the folder now."
        retry_count = 0
        last_action = None
        repeat_action = False
        while not done and step_idx < self.max_steps:

            _, action = self.predict(
                obs
            )
            if action is None:
                logger.info("Failed to parse action from response, try again.")
                retry_count += 1
                if retry_count > 3:
                    logger.info("Failed to parse action from response, stop.")
                    break
                obs = "Failed to parse action from your response, make sure you provide a valid action."
            else:
                logger.info("Step %d: %s", step_idx + 1, action)
                if last_action is not None and last_action == action:
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

    def get_trajectory(self):
        trajectory = []
        for i in range(len(self.observations)):
            trajectory.append({
                "observation": self.observations[i],
                "thought": self.thoughts[i],
                "action": str(self.actions[i]),
                # "code": self.codes[i],
                "response": self.responses[i]
            })
        trajectory_log = {
            "Task": self.instruction,
            "system_message": self.system_message,
            "trajectory": trajectory
        }
        return trajectory_log


if __name__ == "__main__":
    agent = PromptAgent()
    response = """
BIGQUERY_EXEC_SQL(sql_query=\"\"\"
WITH purchase_users AS (
  SELECT DISTINCT user_pseudo_id
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE event_name = 'purchase' AND _TABLE_SUFFIX BETWEEN '20201201' AND '20201231'
),
pageviews AS (
  SELECT user_pseudo_id, COUNT(*) AS pageviews
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE event_name = 'page_view' AND _TABLE_SUFFIX BETWEEN '20201201' AND '20201231'
  GROUP BY user_pseudo_id
),
pageviews_by_user AS (
  SELECT 
    p.user_pseudo_id, 
    p.pageviews,
    CASE WHEN pu.user_pseudo_id IS NOT NULL THEN 'purchaser' ELSE 'non-purchaser' END AS user_type
  FROM pageviews p
  LEFT JOIN purchase_users pu ON p.user_pseudo_id = pu.user_pseudo_id
)
SELECT user_type, AVG(pageviews) AS avg_pageviews
FROM pageviews_by_user
GROUP BY user_type
\"\"\", is_save=True, save_path="avg_pageviews_dec_2020.csv")
"""

    response = """
BIGQUERY_EXEC_SQL(sql_query=\"\"\"
SELECT DISTINCT user_pseudo_id
FROM bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*
WHERE event_name = 'purchase' AND _TABLE_SUFFIX BETWEEN '20201201' AND '20201231'
\"\"\", is_save=False)
"""


    action = agent.parse_action(response)
    print(action)