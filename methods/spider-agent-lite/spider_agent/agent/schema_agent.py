import logging

class SchemaAgentEnv:
    """
    CLI-like environment for schema agent: supports ls, cat, head, grep, etc.
    """
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def step(self, action):
        if hasattr(action, "output"):
            return action.output, True
        # 新增：如果是 Bash 物件，取其 code 屬性
        if hasattr(action, "code"):
            command = action.code
        else:
            command = action
        import subprocess
        try:
            output = subprocess.check_output(command, shell=True, cwd=self.base_dir, text=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            output = e.output
        return output, False

class SchemaAgent:
    """
    An agent that uses LLM to explore a folder (with DDL.csv and other schema files),
    and generates a concise schema string for SQL generation based on user question and critique note.
    """
    def __init__(self, env, llm_predict, max_steps=10, **kwargs):
        self.env = env
        self.max_steps = max_steps
        self.llm_predict = llm_predict

    def run(self, user_question, critique_note):
        obs = self._get_initial_obs(user_question, critique_note)
        result = ""
        done = False
        step_idx = 0
        last_action = None
        repeat_action = False
        while not done and step_idx < self.max_steps:
            _, action = self.llm_predict(obs)
            if action is None:
                obs = "Failed to parse action from your response, make sure you provide a valid action."
                continue
            if last_action is not None and last_action == action:
                if repeat_action:
                    return False, "ERROR: Repeated action"
                else:
                    obs = "The action is the same as the last one, you MUST provide a DIFFERENT CLI command or Terminate action."
                    repeat_action = True
            else:
                obs, done = self.env.step(action)
                last_action = action
                repeat_action = False
            if done:
                if hasattr(action, "output"):
                    result = action.output
                else:
                    result = None  # 或給出明確錯誤訊息
                break
            step_idx += 1
        return result
    def format_schema_prompt(self, schema_dict):
        """
        將 dict 結構轉為 LLM 用 schema string prompt：
        table1(col1[val1,val2], col2[val3,val4]); table2(...)
        """
        prompt_parts = []
        for table, cols in schema_dict.items():
            col_strs = []
            for col, vals in cols.items():
                if vals:
                    val_str = ", ".join(vals[:2])
                    col_strs.append(f"{col}[{val_str}]")
                else:
                    col_strs.append(f"{col}")
            prompt_parts.append(f"{table}({', '.join(col_strs)})")
        return "; ".join(prompt_parts)
    def _get_initial_obs(self, user_question, critique_note):
        return (
            "You are in the folder now.\n"
            "Your goal is to extract a usable schema representation for answering the user's SQL question.\n\n"
            f"User Question:\n{user_question}\n\n"
            f"Critique Note:\n{critique_note}\n\n"
            "Instructions:\n"
            "- Use CLI commands like `ls`, `cat`, `head`, `grep`, etc. to explore the folder.\n"
            "- Look for files like DDL.csv, .json, .csv, or .txt containing schema and sample data.\n"
            "- Only after you’ve collected enough information, return a Terminate action.\n"
            "- Do NOT immediately terminate. You must explore step by step first.\n"
            "- Return the schema in this format:\n"
            '  Terminate(output="table1(col1[val1,val2], col2[val3,val4]); table2(col3[val5])")\n\n'
            "Example final output:\n"
            '  Action: Terminate(output="singer(Singer_ID[1,2], Name[Joe,Timbaland]); song(Song_Name[You], Year[1992])")\n\n'
            "Important:\n"
            "- For each column, try to extract 1–2 sample values from the data.\n"
            "- Do NOT wrap actions in code blocks or quotes. Output your command like:\n"
            '  Action: Bash(code="ls")\n\n'
            "Start with listing the current directory."
        )

