from queries import persdata_query, bestand_query
import pymssql
import os
import pickle
import pandas as pd

from loadcredentials import *

print("load credentials")
print("Enter credential password")
password=input()

dbusr, dbpassword, dbaddress, dbname = read_credentials_from_file(password, ".db_credentials")

print(f"Connect to Database {dbname} at {dbaddress}")

conn = pymssql.connect(dbaddress, dbusr, dbpassword, dbname,as_dict=True)
cursor = conn.cursor()
cursor.execute(persdata_query)

os.makedirs("query_batches", exist_ok=True)
print("Start fetching data")
batch_size = 1000000
i = 1
while True:
    # Fetch batch_size rows at a time
    batch = cursor.fetchmany(batch_size)
    # Break if no more rows
    if not batch:
        break
    # Process the batch
    with open(f"code/query_batches/batch{i}.pkl","wb") as f:
        pickle.dump(batch,f)
    print(f"fetched {batch_size*1} rows")
    i = i+1

conn.close()

print("Aggregate results")
records = []
for batch_file in os.listdir("code/query_batches/"):
    with open(f"code/query_batches/{batch_file}", "rb") as f:
        batch = pickle.load(f)
    records = records + batch

print("Dump result")
persdata = pd.DataFrame(records)
persdata.to_csv("/persdata/persdata.csv",index=False,sep='|')
print(f"Got {persdata.shape[0]} entries")