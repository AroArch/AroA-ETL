import json
import pandas as pd
import re
from collections import defaultdict
import types
from ..utils import re_sub_exclude_parenthesis

def unpack_singleton_lists(col_data):
    """
        Splits list column into separate columns:
        0    [1]           1
        1     3     =>     3
        2    NaN          NaN
    """
    for col in col_data.columns:
        col_data[col] = pd.DataFrame(col_data[col].to_list()).apply(lambda x: x[0] if isinstance(x,list) else x)
    return col_data
    
def split_col_data(col_data):
    """
        Splits list column into separate columns:
        0    [1, 2]           0    1   2
        1       [3]     =>    1    3   NaN
        2    [5, 6]           2    5   6

    """
    for col in col_data.columns:
        split_col_data = pd.DataFrame(col_data[col].to_list())
        split_col_data.columns =  [f"{col}_{i}" for i, _ in enumerate(split_col_data.columns)]
        print(f"Split {col} into {len(split_col_data.columns)} parts")
        col_data = pd.concat([col_data,split_col_data],axis="columns")
        col_data.drop(col,axis="columns",inplace=True)
    return col_data

def flatten_repeat_cell(repeat_cell):
    """
        JSON Data of repeat attributes has the form [{"name": "Alice"}, {"name": "Wonderland"}, ...]
        These entries need to be flattened to a unified dict {"name": ["Alice", "Wonderland"], ...}
    """
    flattened_cell = defaultdict(list)
    for dict_entry in repeat_cell:
        for key, value in dict_entry.items():
            flattened_cell[key].append(str(value).strip())
    return flattened_cell

def filter_na(cell):
    """
        Remove None, redundant and empty values from repeat columns that have only one field, e.g. prisoner_category.
        Not to be used with repeat columns with multiple fields, e.g., dates that have separate day, month and year columns.
    """
    cell = [entry for i, entry in enumerate(cell) if entry not in cell[:i]]
    if len(cell) > 1:
        cell = [entry for entry in cell if entry != None and entry != '']
    if len(cell) == 0:
        cell = ['']
    return cell

def additional_splits(list_of_entries,split_re):
    """
        Split not properly separated values:
        ["Alice, Blice", "Clice"] -> ["Alice", "Blice", "Clice"]
    """
    split_entries = []
    for entry in list_of_entries:
        split_entry = re_sub_exclude_parenthesis(entry.strip(),split_re,"|").split("|")
        #re.sub(split_re,"|",entry.strip()).split("|")
        split_entries = [*split_entries,*split_entry]
    return split_entries

def unpack_col(json_data_df,col,additional_splits_on,split_re):
    print(f"\nstart unpacking of {col}")
    if not re.match(".*_repeat$",col):
        print(f"Unpacking of column without '_repeated' is not implemented")
        return

    # unpack json
    col_data = pd.DataFrame.from_dict(json_data_df[col]\
                                      .apply(flatten_repeat_cell)\
                                      .to_list())
    for inner_col in col_data.columns:
        print(f"  Extracted {inner_col}")
        # do additional splits
        if (isinstance(additional_splits_on, types.FunctionType) and additional_splits_on(inner_col)) \
            or (isinstance(additional_splits_on, list) and inner_col in additional_splits_on):
            print(f"  Applied additional splits")
            col_data[inner_col] = col_data[inner_col].apply(lambda val_list: additional_splits(val_list,split_re))

    # filter redundant and na values for isolated columns, e.g., prisoner_category
    if len(col_data.columns) == 1:
        col_data.iloc[:,0] = col_data.iloc[:,0].apply(filter_na)

    # check for split
    duplicate_value_count = col_data.map(len).max().max()
    if duplicate_value_count > 1:
        col_data = split_col_data(col_data)
    else:
        col_data = unpack_singleton_lists(col_data)

    json_data_df.drop(col,inplace=True,axis="columns")
    return pd.concat([json_data_df,col_data],axis="columns")
    
def unpack(raw_data, json_column, additional_splits_on: 'function | list' = lambda x: False, split_re=r"[\|;,\s]"):
    # parse JSON DataFrame
    json_data_df = pd.DataFrame.from_dict(raw_data[json_column].apply(json.loads).to_list())
    
    # unpack each column
    for col in json_data_df.columns:
        json_data_df = unpack_col(json_data_df,col,additional_splits_on,split_re)

    already_present_columns = set(json_data_df.columns).intersection(set(raw_data.columns))
    assert len(already_present_columns) == 0, f"Unpacking of {json_column} would override columns: {already_present_columns}"
    
    return pd.concat([raw_data.drop(json_column,axis="columns"),json_data_df],axis="columns")