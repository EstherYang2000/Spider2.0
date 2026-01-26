"""
Schema Agent Module

This module implements the SchemaAgent and SchemaAgentEnv classes for automated
database schema exploration. The agent uses language models to explore schema files
and generate concise schema representations for SQL query generation.
"""



class SchemaAgentEnv:
    """
    CLI-like environment for schema exploration.

    Provides a command-line interface environment that supports basic shell commands
    like ls, cat, head, grep, etc. for exploring database schema files and directories.
    Used by the SchemaAgent to interact with the file system during schema discovery.
    """

    def __init__(self, base_dir):
        """
        Initialize the schema agent environment.

        Args:
            base_dir: Base directory path where schema files are located.
        """
        self.base_dir = base_dir

    def step(self, action):
        """
        Execute a CLI command in the environment.

        Supports various action types including direct commands and action objects
        with output or code attributes. Executes shell commands and returns output.

        Args:
            action: Command string or action object to execute.

        Returns:
            tuple: (output, done) where output is command result and done indicates completion.
        """
        if hasattr(action, "output"):
            return action.output, True
        # Handle Bash action objects with code attribute
        if hasattr(action, "code"):
            command = action.code
        else:
            command = action
        import subprocess

        try:
            output = subprocess.check_output(
                command,
                shell=True,
                cwd=self.base_dir,
                text=True,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            output = e.output
        return output, False


class SchemaAgent:
    """
    Agent for automated database schema exploration using language models.

    This agent uses LLM guidance to explore database schema files (DDL.csv, JSON, etc.)
    through CLI commands and generates concise schema representations for SQL query generation.
    It iteratively explores the schema directory based on user questions and critique feedback.
    """

    def __init__(self, env, llm_predict, max_steps=20, **kwargs):
        """
        Initialize the SchemaAgent.

        Args:
            env: SchemaAgentEnv instance for executing CLI commands.
            llm_predict: Function for getting LLM predictions/actions.
            max_steps: Maximum number of exploration steps allowed.
            **kwargs: Additional keyword arguments.
        """
        self.env = env
        self.max_steps = max_steps
        self.llm_predict = llm_predict

    def run(self, user_question, critique_note):
        """
        Run the schema exploration process.

        Uses iterative LLM-guided exploration to discover and understand the database schema,
        preventing action repetition and enforcing step limits.

        Args:
            user_question: The user's SQL question to guide schema exploration.
            critique_note: Additional critique or context information.

        Returns:
            str or tuple: Extracted schema string, or error tuple if exploration fails.
        """
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
                    result = None  # Provide explicit error message
                break
            step_idx += 1
        return result

    def format_schema_prompt(self, schema_dict):
        """
        Format schema dictionary into LLM prompt string.

        Converts a dictionary representation of database schema into a concise
        string format suitable for SQL generation prompts.

        Args:
            schema_dict: Dictionary with table names as keys and column info as values.

        Returns:
            str: Formatted schema string in the format "table1(col1[val1,val2], col2[val3,val4]); table2(...)"
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
        """
        Generate initial observation/prompt for schema exploration.

        Creates the starting prompt that guides the LLM in exploring the schema
        directory and understanding what information to extract.

        Args:
            user_question: The user's SQL question.
            critique_note: Additional critique or context information.

        Returns:
            str: Initial observation prompt with instructions for schema exploration.
        """
        return (
            "You are in the folder now.\n"
            "Your goal is to extract a usable schema representation for answering the user's SQL question.\n\n"
            f"User Question:\n{user_question}\n\n"
            f"Critique Note:\n{critique_note}\n\n"
            "Instructions:\n"
            "- Use CLI commands like `ls`, `cat`, `head`, `grep`, etc. to explore the folder.\n"
            "- Look for files like DDL.csv, .json, .csv, or .txt containing schema and sample data.\n"
            "- Only after you've collected enough information, return a Terminate action.\n"
            "- Do NOT immediately terminate. You must explore step by step first.\n"
            "- Return the schema in this format:\n"
            '  Terminate(output="table1(col1:TYPE[val1,val2], col2:TYPE[val3,val4]); table2(col3:TYPE[val5])")\n\n'
            "Example final output:\n"
            '  Action: Terminate(output="singer(Singer_ID:INT[1,2], Name:VARCHAR[Joe,Timbaland]); song(Song_Name:VARCHAR[You], Year:INT[1992])")\n\n'
            "Important:\n"
            "- For each column, try to extract 1–2 sample values from the data.\n"
            "- Include the data type for each column (e.g., INT, VARCHAR, TIMESTAMP, VARIANT, etc.).\n"
            "- For Snowflake, use these common types:\n"
            "  * NUMBER/INT/FLOAT for numeric values\n"
            "  * VARCHAR/STRING for text\n"
            "  * TIMESTAMP for dates and times\n"
            "  * VARIANT for JSON/array data\n"
            "  * BOOLEAN for true/false values\n"
            "- Do NOT wrap actions in code blocks or quotes. Output your command like:\n"
            '  Action: Bash(code="ls")\n\n'
            "Start with listing the current directory."
        )
