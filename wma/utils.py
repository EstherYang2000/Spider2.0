import json
import re
import pandas as pd
import math
from typing import List, Union
import os
import pandas as pd
from google.cloud import bigquery
import sqlite3
from tqdm import tqdm
import snowflake.connector

def get_bigquery_sql_result(sql_query):
    """
    is_save = True, output a 'result.csv'
    if_save = False, output a string
    """
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "methods/spider-agent-lite/bigquery_credential.json"
    client = bigquery.Client()


    try:
        query_job = client.query(sql_query)
        results = query_job.result().to_dataframe()         
        return results
        
        # if results.empty:
        #     print("No data found for the specified query.")
        #     # results.to_csv(os.path.join(save_dir, file_name), index=False)
        #     return False, None
        # else:
        #     if is_save:
        #         results.to_csv(os.path.join(save_dir, file_name), index=False)
        #         return True, None
        #     else:
        #         value = results.iat[0, 0]
        #         return True, None
    except Exception as e:
        print("Error occurred while fetching data: ", e)  
        return None

# Fetch SQL query results from Snowflake
def get_snowflake_sql_result(sql_query, database_id):
    """
    is_save = True, output a 'result.csv'
    if_save = False, output a string
    """
    snowflake_credential = json.load(open('methods/spider-agent-lite/snowflake_credential.json'))
    conn = snowflake.connector.connect(
        database=database_id,
        **snowflake_credential
    )
    cursor = conn.cursor()
    
    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(results, columns=columns)
        # if df.empty:
        #     print("No data found for the specified query.")
        #     return False, None
        # else:
        #     if is_save:
        #         df.to_csv(os.path.join(save_dir, file_name), index=False)
        #         return True, None
        return df
    except Exception as e:
        print("Error occurred while fetching data: ", e)  
        return None

# Fetch SQLite query results
def get_sqlite_result(db_path, query):
    conn = sqlite3.connect(db_path)
    memory_conn = sqlite3.connect(':memory:')

    conn.backup(memory_conn)
    
    try:
        # if save_dir:
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     for i, chunk in enumerate(pd.read_sql_query(query, memory_conn, chunksize=chunksize)):
        #         mode = 'a' if i > 0 else 'w'
        #         header = i == 0
        #         chunk.to_csv(os.path.join(save_dir, file_name), mode=mode, header=header, index=False)
        # else:
        df = pd.read_sql_query(query, memory_conn)
            # return True, df
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        # return False, str(e)
        return None
    finally:
        memory_conn.close()
        conn.close()
    
def append_json(file_path, new_data):
    """ Append new data to an existing JSON file or create a new one if it doesn't exist. """
    # Check if file exists and read existing content
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]  # Ensure it's a list
            except json.JSONDecodeError:
                existing_data = []  # If file is empty or corrupt, start fresh
    else:
        existing_data = []

    # Append new data
    if isinstance(new_data, list):
        existing_data.extend(new_data)
    else:
        existing_data.append(new_data)

    # Write back to file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)


# Compare two pandas tables based on conditions
def compare_pandas_table(pred, gold, condition_cols=[], ignore_order=False):
    """_summary_

    Args:
        pred (Dataframe): _description_
        gold (Dataframe): _description_
        condition_cols (list, optional): _description_. Defaults to [].
        ignore_order (bool, optional): _description_. Defaults to False.

    """
    # print('condition_cols', condition_cols)
    
    tolerance = 1e-2

    def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
        if ignore_order_:
            v1, v2 = (sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))),
                    sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))))
        if len(v1) != len(v2):
            return False
        for a, b in zip(v1, v2):
            if pd.isna(a) and pd.isna(b):
                continue
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), abs_tol=tol):
                    return False
            elif a != b:
                return False
        return True
    
    if condition_cols != []:
        gold_cols = gold.iloc[:, condition_cols]
    else:
        gold_cols = gold
    pred_cols = pred

    t_gold_list = gold_cols.transpose().values.tolist()
    t_pred_list = pred_cols.transpose().values.tolist()
    score = 1
    for _, gold in enumerate(t_gold_list):
        if not any(vectors_match(gold, pred, ignore_order_=ignore_order) for pred in t_pred_list):
            score = 0
        else:
            for j, pred in enumerate(t_pred_list):
                if vectors_match(gold, pred, ignore_order_=ignore_order):
                    break

    return score
# Compare multiple pandas tables to check if at least one match exists
def compare_multi_pandas_table(pred, multi_gold, multi_condition_cols=[], multi_ignore_order=False):
    # print('multi_condition_cols', multi_condition_cols)

    if multi_condition_cols == [] or multi_condition_cols == [[]] or multi_condition_cols == [None] or multi_condition_cols == None:
        multi_condition_cols = [[] for _ in range(len(multi_gold))]
    elif len(multi_gold) > 1 and not all(isinstance(sublist, list) for sublist in multi_condition_cols):
        multi_condition_cols = [multi_condition_cols for _ in range(len(multi_gold))]
    multi_ignore_order = [multi_ignore_order for _ in range(len(multi_gold))]

    for i, gold in enumerate(multi_gold):
        if compare_pandas_table(pred, gold, multi_condition_cols[i], multi_ignore_order[i]):
            return 1
    return 0
def load_jsonl_to_dict(jsonl_file):
    data_dict = {}
    with open(jsonl_file, 'r') as file:
        for line in file:
            item = json.loads(line.strip())
            instance_id = item['instance_id']
            data_dict[instance_id] = item
    return data_dict
def pre_evaluate_spider2sql(mode: str, gold_data: pd.DataFrame, pred_data: pd.DataFrame, id: str):
    eval_standard_dict = load_jsonl_to_dict(os.path.join("spider2-lite/evaluation_suite/gold", "spider2lite_eval.jsonl"))
    spider2sql_metadata = load_jsonl_to_dict("spider2-lite/spider2-lite.jsonl")
    if mode == "sql":
        pred_sql_query = open(os.path.join(pred_result_dir, f"{id}.sql")).read()

    elif mode == "exec_result":
        try:
            print("exec_result......")
            print(pred_data)
            print(gold_data)
            score = compare_multi_pandas_table(pred_data, gold_data, eval_standard_dict.get(id)['condition_cols'], eval_standard_dict.get(id)['ignore_order'])
            return score
        except Exception as e:
            print(f"ERROR: {str(e)}")
            return 0