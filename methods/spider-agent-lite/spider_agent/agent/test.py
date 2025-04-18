from google.oauth2 import service_account
from google.cloud import bigquery

credential_path = 'bigquery_credential.json' # path/to/your/keyfile.json
credentials = service_account.Credentials.from_service_account_file(credential_path)
client = bigquery.Client(credentials=credentials)

# alternatively, you can also set the credential path via environment vairable
# import os
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/path/to/keyfile.json"
# client = bigquery.Client()

# Perform a sample query.
sql_query = 'SELECT name FROM `bigquery-public-data.usa_names.usa_1910_2013` WHERE state = "TX" LIMIT 10'
query_job = client.query(sql_query)  # API request
rows = query_job.result()  # Waits for query to finish

for row in rows:
    print(row.name)