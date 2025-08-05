import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series
import numpy as np
import sys

input_fname = sys.argv[1]
fname, ftype = input_fname.split(".")
out_fname = f"{fname}_deduplicated"


data = pd.read_csv(input_fname, sep="|") if ftype == "csv" else pd.read_excel(input_fname)

deduplication_keys = ["Objekt ID", "Count ID"]

def remove_duplicates(values: Series):
    return [v for i, v in enumerate(values) if v not in values.values[:i] and pd.notna(v)]

def deduplicate_group(group_keys: tuple, group: DataFrame):
    deduplicated_data = {
        col: ";".join(map(str,remove_duplicates(group[col])))
        for col in group.columns
    }
    deduplicated_data_df = pd.Series(deduplicated_data).to_frame().T
    deduplicated_data_df[deduplication_keys] = group_keys
    return deduplicated_data_df

groups = data.groupby(deduplication_keys)

deduplicated_data_df_list = []

for i, (group_keys, group) in enumerate(groups):
    deduplicated_data_df = deduplicate_group(group_keys, group)
    deduplicated_data_df_list.append(deduplicated_data_df)

deduplicated_data_df = pd.concat(deduplicated_data_df_list, ignore_index=True, axis=0)
deduplicated_data_df.to_csv(f"{out_fname}.csv", sep="|")
deduplicated_data_df.to_excel(f"{out_fname}.xlsx")
