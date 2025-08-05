import pandas as pd
import re
from ..utils import split_cols_by_basename, split_dmy_date_cols, recombine_col_split
import aroa_etl.attribute_processing.column_processing as cp


# --------------- Processing of Columns ------------------------
def process_last_name_columns(df, last_name_columns: list,
                        data_source="crowd_sourcing",
                        birth_name_input=False,
                        alternative_spelling=False,
                        alias=False,
                        flag_noble_prefix=True):
    """
        This function, first, clusters last_name columns to entities [person_last_name_columns, relative_last_name_columns, ...]
        and, second, applies standard column processing and normalizing to the each entities last name column.
        Returns a DataFrame containing the processed last names, qa flags, data source and, optionally, extracted birthnames and aliases.
    """
    entity_name_df_list = []
    for entity_cols in split_cols_by_basename(last_name_columns):  
        entity_slice = recombine_col_split(df[entity_cols])
        entity_name_df = cp.normalize_last_name(entity_slice,
                        last_name=entity_slice.columns[0],
                        data_source=data_source,
                        birth_name_input=birth_name_input,
                        alternative_spelling=alternative_spelling,
                        alias=alias,
                        flag_noble_prefix=flag_noble_prefix)
        entity_name_df_list.append(entity_name_df)
    entity_name_df = None
    if len(entity_name_df_list)>0:
        entity_name_df = pd.concat(entity_name_df_list, axis="columns")
    return entity_name_df

def process_first_name_columns(df, first_name_columns: list,
                        data_source="crowd_sourcing",
                        alternative_spelling=False,
                        alias=False,):
    """
        This function, first, clusters first_name columns to entities [person_first_name_columns, relative_first_name_columns, ...]
        and, second, applies standard column processing and normalizing to the each entities first name column.
        Returns a DataFrame containing the processed last names, qa flags, data source and, optionally, extracted aliases.
    """
    entity_name_df_list = []
    for entity_cols in split_cols_by_basename(first_name_columns):  
        entity_slice = recombine_col_split(df[entity_cols])
        entity_name_df = cp.normalize_first_name(entity_slice,
                        first_name=entity_slice.columns[0],
                        data_source=data_source,
                        alternative_spelling=alternative_spelling,
                        alias=alias,)
        entity_name_df_list.append(entity_name_df)
    entity_name_df = None
    if len(entity_name_df_list)>0:
        entity_name_df = pd.concat(entity_name_df_list, axis="columns")
    return entity_name_df


def process_prisoner_number_columns(df, prisoner_number_columns,
                                   data_source="crowd_sourcing"):
    """
        This applies processing, normalization and qa flagging to all prisoner number columns.
    """
    entity_name_df_list = []
    for entity_cols in split_cols_by_basename(prisoner_number_columns):  
        entity_slice = recombine_col_split(df[entity_cols])
        entity_name_df = cp.normalise_prisoner_number(entity_slice,
                              prisoner_no=entity_slice.columns[0],
                              data_source=data_source)
        entity_name_df_list.append(entity_name_df)
    entity_name_df = None
    if len(entity_name_df_list)>0:
        entity_name_df = pd.concat(entity_name_df_list, axis="columns")
    return entity_name_df

def process_date_columns(df, date_columns,
                         data_source="crowd_sourcing",
                         timespan=[1850,1950]):
    """
        This function, first, clusters separate day, month, year columns to dates [[dob_day, dob_month, dob_year], ...]
        and, second, applies standard column processing and normalizing to the each date.
    """
    date_df_list = []
    for entity_cols in split_dmy_date_cols(date_columns):
        # present columns in day, month, year order
        assert len(entity_cols) == 3, f"Date columns should only include day, month and year in \n {entity_cols}"
        day_col = filter(lambda col: "_day" in col, entity_cols).__next__()
        month_col = filter(lambda col: "_month" in col, entity_cols).__next__()
        year_col = filter(lambda col: "_year" in col, entity_cols).__next__()

        output_name = re.sub(r"(?P<dmy>_year)(?P<num>_\d+)?$",r"\g<num>",year_col)
        
        date_df = cp.clean_date(df, date_col_list=[day_col, month_col, year_col],
                                timespan=[1850, 1950],
                                output_name=output_name,
                                dropdown=False,
                                data_source=data_source)
        date_df_list.append(date_df)
    date_df = None
    if len(date_df_list)>0:
        date_df = pd.concat(date_df_list, axis="columns")
    return date_df

def process_raw_date_columns(df, raw_date_columns,
                             data_source="crowd_sourcing",
                             timespan=[1850,1950]):
    """
        This applies processing, normalization and qa flagging to all date columns,
        where day, month and year are in one cell.
    """
    date_df_list = []
    for col in raw_date_columns:
        date_df = cp.normalise_date(df,
                                    date_col_name=col,
                                    date_output_name=col,
                                    data_source=data_source,
                                    timespan=timespan)
        date_df_list.append(date_df)
    date_df = None
    if len(date_df_list)>0:
        date_df = pd.concat(date_df_list, axis="columns")
    return date_df

def process_location_columns(df, location_columns,
                              data_source="crowd_sourcing",):
    """
        This function applies default processing and qa flagging to all `other_columns`.
    """
    location_df_list = []
    for col in location_columns:
        print("WARNING: Location processor is not implemented. Use default processor instead.")
        location_df = cp.default_processor(df, column=col,
                                           data_source=data_source)
        location_df_list.append(location_df)
    location_df = None
    if len(location_df_list)>0:
        location_df = pd.concat(location_df_list, axis="columns")
    return location_df

def process_nationality_columns(df, nationality_columns,
                                data_source="crowd_sourcing",):
    """
        This function, first, clusters first_name columns to entities [person_first_name_columns, relative_first_name_columns, ...]
        and, second, applies standard column processing and normalizing to the each entities first name column.
        Returns a DataFrame containing the processed last names, qa flags, data source and, optionally, extracted aliases.
    """
    nationality_df_list = []
    for col in nationality_columns:
        # TODO: relies on external data and no data_source is used. This is inconsistent to the other columns
        # nationality_df = cp.standardize_nationality(df,
        #                                     nat_column=col,
        #                                     path="TODO",
        #                                     nat_dict_table="TODO",
        #                                     unclear_table="TODO",)
        print("WARNING: Nationality processor is not implemented. Use default processor instead.")
        nationality_df = cp.default_processor(df, column=col,
                                              data_source=data_source)
        nationality_df_list.append(nationality_df)
    nationality_df = None
    if len(nationality_df_list)>0:
        nationality_df = pd.concat(nationality_df_list, axis="columns")
    return nationality_df

def process_other_columns(df, other_columns,
                              data_source="crowd_sourcing",):
    """
        This function applies default processing and qa flagging to all `other_columns`.
    """
    other_df_list = []
    for col in other_columns:
        other_df = cp.default_processor(df, column=col,
                                        data_source=data_source)
        other_df_list.append(other_df)
    other_df = None
    if len(other_df_list)>0:
        other_df = pd.concat(other_df_list, axis="columns")
    return other_df

# --------------- Clusteres Columns by type ------------------------

def automatic_column_type_detection(df,
                                    last_name_columns: list=None, 
                                    first_name_columns: list=None, 
                                    prisoner_number_columns: list=None,
                                    date_columns: list=None,
                                    raw_date_columns: list=None,
                                    location_columns: list=None, 
                                    nationality_columns: list=None,
                                    skip_columns: list=[],):
    """
        This function automatically clusters the columns of a dataframe to fields that have specific processing routine.
        (e.g., Last Names, First Names, Dates, ...)
        The default detection can be overwritten by this functions arguments. 
    """
    other_columns = set(df.columns)

    if last_name_columns is None:
        last_name_columns = {col for col in other_columns if "last" in re.findall("[a-zA-Z]*",col) and "name" in re.findall("[a-zA-Z]*",col)}
        print(f"Last Name Columns:\n{"\n".join(sorted(last_name_columns))}\n")
    other_columns = other_columns.difference(last_name_columns)
    
    if first_name_columns is None:
        first_name_columns = {col for col in other_columns if "first" in re.findall("[a-zA-Z]*",col) and "name" in re.findall("[a-zA-Z]*",col)}
        print(f"First Name Columns:\n{"\n".join(sorted(first_name_columns))}\n")
    other_columns = other_columns.difference(first_name_columns)

    if prisoner_number_columns is None:
        prisoner_number_columns = {col for col in other_columns if "prisoner" in re.findall("[a-zA-Z]*",col) and "number" in re.findall("[a-zA-Z]*",col)}
        print(f"Prisoner Number Columns:\n{"\n".join(sorted(prisoner_number_columns))}\n")
    other_columns = other_columns.difference(prisoner_number_columns)
    
    if date_columns is None:
        date_columns = {col for col in other_columns if "day" in re.findall("[a-zA-Z]*",col) or "month" in re.findall("[a-zA-Z]*",col) or "year" in re.findall("[a-zA-Z]*",col)}
        print(f"Date Columns:\n{"\n".join(sorted(date_columns))}\n")
    other_columns = other_columns.difference(date_columns)

    if raw_date_columns is None:
        raw_date_columns = {col for col in other_columns if "date" in re.findall("[a-zA-Z]*",col)}
        print(f"Raw Date Columns:\n{"\n".join(sorted(raw_date_columns))}\n")
    other_columns = other_columns.difference(raw_date_columns)

    if location_columns is None:
        location_columns = {col for col in other_columns if "place" in re.findall("[a-zA-Z]*",col)}
        print(f"Location Columns:\n{"\n".join(sorted(location_columns))}\n")
    other_columns = other_columns.difference(location_columns)

    if nationality_columns is None:
        nationality_columns = {col for col in other_columns if "nation" in re.findall("[a-zA-Z]*",col)}
        print(f"Nationality Columns:\n{"\n".join(sorted(nationality_columns))}\n")
    other_columns = other_columns.difference(nationality_columns)
    # do not process columns in skip_columns
    other_columns = other_columns.difference(skip_columns)
    
    print(f"Other Columns:\n{"\n".join(sorted(other_columns))}\n")
    return last_name_columns, first_name_columns, prisoner_number_columns, date_columns, raw_date_columns, location_columns, nationality_columns, other_columns

# ------------------- Default Processing -----------------------

def apply_split_limit(df, split_limit: int):
    col_number = lambda col: re.search(r"\d+$",col)
    below_limit = lambda d: int(d.group())<=split_limit
    return df[[col for col in df.columns if not col_number(col) or below_limit(col_number(col))]]

def process_unpacked_data(df,
                          last_name_columns: list=None, 
                          first_name_columns: list=None, 
                          prisoner_number_columns: list=None,
                          date_columns: list=None,
                          raw_date_columns: list=None,
                          location_columns: list=None, 
                          nationality_columns: list=None, 
                          other_columns: list=None,
                          skip_columns: list=[],
                          include_data_source = True,
                          include_original = True,
                          split_limit: int = 5,
                         ):
    """
        This function applies a default processing and normalization to df.
    """
    print("Fill na with '' and convert Data to String")
    df = df.fillna('').astype(str)

    # remove all columns above the limit, e.g., first_name_17
    df = apply_split_limit(df, split_limit)
    
    automatic_columns = automatic_column_type_detection(df,
                                    last_name_columns=last_name_columns, 
                                    first_name_columns=first_name_columns, 
                                    prisoner_number_columns=prisoner_number_columns,
                                    date_columns=date_columns,
                                    raw_date_columns=raw_date_columns,
                                    location_columns=location_columns, 
                                    nationality_columns=nationality_columns,
                                    skip_columns=skip_columns)
    last_name_columns, first_name_columns, prisoner_number_columns, date_columns, raw_date_columns, location_columns, nationality_columns, other_columns = automatic_columns
    print(f"Start processing of last name columns")
    last_name_df = process_last_name_columns(df, last_name_columns)

    print(f"Start processing of first name columns")
    first_name_df = process_first_name_columns(df, first_name_columns)

    print(f"Start processing prisoner numbers")
    prisoner_number_df = process_prisoner_number_columns(df, prisoner_number_columns)

    print(f"Start processing dates")
    date_df = process_date_columns(df, date_columns)

    print(f"Start processing raw dates")
    raw_date_df = process_raw_date_columns(df, raw_date_columns)

    print(f"Start processing locations")
    location_df = process_location_columns(df, location_columns)
    
    print(f"Start processing nationalities")
    nationality_df = process_nationality_columns(df, nationality_columns)
    
    print(f"Start processing other columns")
    other_df = process_other_columns(df, other_columns)
    df = pd.concat([
        df if include_original else None,
        last_name_df,
        first_name_df,
        prisoner_number_df,
        date_df,
        raw_date_df,
        location_df,
        nationality_df,
        other_df, ], axis="columns")
    if not include_data_source:
        df = df[[col for col in df.columns if "data_source" not in col]]
    return df