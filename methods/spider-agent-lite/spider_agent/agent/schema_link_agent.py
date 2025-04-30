from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import os
import json

class SchemaLinkAgent:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2', mask_token=None, value_token=None):
        self.model = SentenceTransformer(model_name)
        self.mask_token = mask_token
        self.value_token = value_token

    def mask_question(self, question, schema_items):
        # 可根據需求設計 mask 規則，這裡簡單替換 schema 名稱為 <mask>
        q = question
        for item in schema_items:
            q = q.replace(item, self.mask_token or "<mask>")
        return q

    def link(self, question, schema, data_path, top_k_tables=5, top_k_columns=20):
        """
        question: str
        schema: pd.DataFrame (with columns: table_name, DDL)
        data_path: 資料夾路徑，內含 DDL.csv 與每個 table 的 json
        """
        import faiss
        import numpy as np

        # 取得所有 table 名稱
        tables = schema['table_name'].unique().tolist()
        table_descs = [""] * len(tables)  # 可根據需求補上描述
        # 產生 table embedding
        table_texts = [f"{t}: {d}" for t, d in zip(tables, table_descs)]
        table_embs = self.model.encode(table_texts, normalize_embeddings=True)
        table_index = faiss.IndexFlatIP(table_embs.shape[1])
        table_index.add(np.array(table_embs, dtype=np.float32))

        # 產生 column embedding
        columns = []
        column_texts = []
        column_table_map = []
        for table in tables:
            json_path = os.path.join(data_path, f"{table}.json")
            if not os.path.exists(json_path):
                continue
            with open(json_path, 'r') as f:
                meta = json.load(f)
            cols = meta.get('nested_column_names', meta.get('column_names', []))
            for col in cols:
                columns.append({"table": table, "name": col, "desc": ""})
                column_texts.append(f"{table}.{col}: ")
                column_table_map.append(table)
        if not tables or not columns:
            return {"linked_tables": [], "linked_columns": []}

        # 語意檢索 table
        q_emb = self.model.encode([question], normalize_embeddings=True)
        D, I = table_index.search(q_emb, min(top_k_tables, len(tables)))
        selected_tables = [tables[i] for i in I[0]]

        # 語意檢索 column（僅在 top tables 下）
        candidate_columns = [i for i, t in enumerate(column_table_map) if t in selected_tables]
        if not candidate_columns:
            return {"linked_tables": selected_tables, "linked_columns": []}
        column_embs = self.model.encode([column_texts[i] for i in candidate_columns], normalize_embeddings=True)
        probe_index = faiss.IndexFlatIP(column_embs.shape[1])
        probe_index.add(np.array(column_embs, dtype=np.float32))
        D2, I2 = probe_index.search(q_emb, min(top_k_columns, len(candidate_columns)))
        selected_columns = [columns[candidate_columns[i]] for i in I2[0]]

        # 回傳 linked_tables/linked_columns
        linked_tables = list(set([col['table'] for col in selected_columns] + selected_tables))
        linked_columns = [f"{col['table']}.{col['name']}" for col in selected_columns]

        return {"linked_tables": linked_tables, "linked_columns": linked_columns}

    def find_ddl_csv(self, example_dir, db):
        for root, dirs, files in os.walk(example_dir):
            for file in files:
                if file.lower() == "ddl.csv" and db in root:
                    return (os.path.join(root, file), root)
        return None

    def get_sample_rows(self,root, tables, max_rows=3):
        sample_rows = {}
        for table in tables:
            sample_file_json = os.path.join(root, f"{table}.json")
            rows = []
            if os.path.exists(sample_file_json):
                try:
                    with open(sample_file_json, 'r') as f:
                        data = json.load(f)
                        # Try to get a list of rows; adapt this if your JSON structure is different
                        if isinstance(data, list):
                            rows = data[:max_rows]
                        elif isinstance(data, dict) and "rows" in data:
                            rows = data["rows"][:max_rows]
                        elif isinstance(data, dict) and "sample_rows" in data:
                            rows = data["sample_rows"][:max_rows]
                except Exception as e:
                    print(f"Failed to load sample rows for table {table} from JSON: {e}")
            sample_rows[table] = rows
        return sample_rows

    def get_sqlite_sample_rows(self,sqlite_path, tables, max_rows=3):
        import sqlite3
        sample_rows = {}
        try:
            conn = sqlite3.connect(sqlite_path)
            conn.row_factory = sqlite3.Row  # Fetch rows as dictionaries
            cursor = conn.cursor()
            for table in tables:
                try:
                    cursor.execute(f"SELECT * FROM '{table}' LIMIT {max_rows}")
                    rows = [dict(row) for row in cursor.fetchall()]
                    sample_rows[table] = rows
                except Exception as e:
                    print(f"Failed to fetch sample rows for table {table} from sqlite: {e}")
                    sample_rows[table] = []
            conn.close()
        except Exception as e:
            print(f"Failed to connect to sqlite database: {e}")
        return sample_rows

if __name__ == "__main__":
    agent = SchemaLinkAgent()
    question = "For patents granted between 2010 and 2018, provide the publication number of each patent and the number of backward citations it has received in the SEA category."
    schema = pd.read_csv("output/grok-3-beta-test8-plan-self-refinement/sf_bq027/PATENTS/PATENTS/DDL.csv")
    data_path = "output/grok-3-beta-test8-plan-self-refinement/sf_bq027/PATENTS/PATENTS"
    result = agent.link(question, schema, data_path, top_k_tables=5, top_k_columns=20)
    print(result)    
    # EXAMPLES_DIR = "examples"
    # INPUT_JSONL = os.path.join(EXAMPLES_DIR, "spider2-lite.jsonl")
    # OUTPUT_JSONL = os.path.join(EXAMPLES_DIR, "spider2-lite_sl.jsonl")

    # # Test
    # agent = SchemaLinkAgent()
    # with open(INPUT_JSONL, "r") as f:
    #     lines = f.readlines()
    # new_lines = []
    # from tqdm import tqdm
    # for line in tqdm(lines):
    #     item = json.loads(line)
    #     instance_id = item.get("instance_id")
    #     db = item.get("db")
    #     question = item.get("question")
    #     if not (instance_id and db and question):
    #         item["schema_link"] = None
    #         new_lines.append(json.dumps(item, ensure_ascii=False))
    #         continue
    #     # local: 用 sqlite 取 schema
    #     if instance_id.startswith("local"):
    #         sqlite_path = os.path.join(EXAMPLES_DIR, instance_id, f"{db}.sqlite")
    #         if os.path.exists(sqlite_path):
    #             try:
    #                 import sqlite3
    #                 conn = sqlite3.connect(sqlite_path)
    #                 cursor = conn.cursor()
    #                 cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    #                 tables = [row[0] for row in cursor.fetchall()]
    #                 schema_rows = []
    #                 for table in tables:
    #                     cursor.execute(f"PRAGMA table_info('{table}')")
    #                     for col in cursor.fetchall():
    #                         schema_rows.append({
    #                             "table_name": table,
    #                             "column_name": col[1],
    #                             "type": col[2]
    #                         })
    #                 conn.close()
    #                 ddl = pd.DataFrame(schema_rows)
    #                 print(f"[{instance_id}] Schema linking from sqlite...")
    #                 print(ddl.head(1))
    #                 result = agent.link(question, ddl, os.path.dirname(sqlite_path))
    #                 print(result)
    #                 item["schema_link"] = result
    #                 # Add sample rows for linked tables
    #                 linked_tables = result.get("linked_tables", [])
    #                 item["sample_rows"] = agent.get_sqlite_sample_rows(sqlite_path, linked_tables)
    #             except Exception as e:
    #                 print(f"[{instance_id}] Schema linking failed (sqlite): {e}")
    #                 item["schema_link"] = None
    #         else:
    #             print(f"[{instance_id}] sqlite not found: {sqlite_path}")
    #             item["schema_link"] = None
    #     else:
    #         # 非 local: 用 DDL.csv
    #         example_dir = os.path.join(EXAMPLES_DIR, instance_id)
    #         ddl_path, root = agent.find_ddl_csv(example_dir, db)
    #         if ddl_path and os.path.exists(ddl_path):
    #             try:
    #                 ddl = pd.read_csv(ddl_path)
    #                 print(f"[{instance_id}] Schema linking...")
    #                 print(ddl.head(1))
    #                 if ddl.empty:
    #                     print(f"[{instance_id}] DDL is empty, skipping schema linking.")
    #                     item["schema_link"] = None
    #                 else:
    #                     result = agent.link(question, ddl, root)
    #                     print(result)
    #                     item["schema_link"] = result
    #                     # Add sample rows for linked tables
    #                     linked_tables = result.get("linked_tables", [])
    #                     item["sample_rows"] = agent.get_sample_rows(root, linked_tables)
    #             except Exception as e:
    #                 print(f"[{instance_id}] Schema linking failed: {e}")
    #                 item["schema_link"] = None
    #         else:
    #             print(f"[{instance_id}] DDL.csv not found for db={db}")
    #             item["schema_link"] = None

    #         new_lines.append(json.dumps(item, ensure_ascii=False))

    # with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    #     for l in new_lines:
    #         f.write(l + "\n")
    # print(f"Done! Output written to {OUTPUT_JSONL}")
