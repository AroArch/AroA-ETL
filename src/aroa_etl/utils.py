import pandas as pd
from collections.abc import Iterable
import re

NA_VALUES = ["-1", "-1.0", "None", "", "NULL", "unbekannt", "unbekant", "-", "0", "0.0", "NA", "00", "0000", ]
QA_VALUES = ["?", "unklar", "Unklar"]

def value_is_empty_q(val):
    """
        Returns true if a value does not contain information:
        - ignores empty strings
        - The '-' symbol
        - Empty iterables
    """
    empty_string = lambda v: True if v in NA_VALUES else False
    return True \
           if (isinstance(val, Iterable) and not isinstance(val, str) and len([v for v in val if not empty_string(v)]) == 0) \
           or (isinstance(val, str) and empty_string(val)) \
           or (pd.isna(val) or val is None)\
           else False

def value_is_not_empty_q(val):
    """
        Returns true if a value does contain information:
        - ignores empty strings
        - The '-' symbol
        - Empty iterables
    """
    return not value_is_empty_q(val)

def __has_no_value_q(val):
    empty_string = lambda v: True if v in NA_VALUES + QA_VALUES else False
    return True \
           if (isinstance(val, Iterable) and not isinstance(val, str) and len([v for v in val if not empty_string(v)]) == 0) \
           or (isinstance(val, str) and empty_string(val)) \
           or (pd.isna(val) or val is None)\
           else False

def has_value_q(val):
    return not __has_no_value_q(val)

def re_sub_exclude_parenthesis(string, pattern, repl):
    """
        String replace wiht regular expression, but ignore text within parenthesis. Supports nesting.
    """
    string = re.sub(r"(?P<par>[\(\)\[\]])",r"<SPLIT>\g<par><SPLIT>",string)
    allow_sub = 0
    result = ""
    for sub_str in string.split("<SPLIT>"):
        if re.match(r"[\(\[]",sub_str):
            allow_sub += 1
        elif re.match(r"[\)\]]",sub_str):
            allow_sub -= 1
        elif allow_sub == 0:
            sub_str = re.sub(pattern, repl, sub_str)
        result += sub_str
    return result

def remove_parenthesis_substr(string):
    """
        Removes substring that is enclosed in parenthesis. Supports nesting.
    """
    string = re.sub(r"(?P<par>[\(\)\[\]])",r"<SPLIT>\g<par><SPLIT>",string)
    allow_sub = 0
    result = ""
    for sub_str in string.split("<SPLIT>"):
        if re.match(r"[\(\[]",sub_str):
            allow_sub += 1
        elif re.match(r"[\)\]",sub_str):
            allow_sub -= 1
        elif allow_sub == 0:
            sub_str = re.sub(pattern, repl, sub_str)
            result += sub_str
    return result

# -----------------------   Pandas ------------------------------------

def split_dmy_date_cols(cols):
    """
        This function clusters a list of date columns into: 
            [[birthday_day_0, birthday_month_0, birthday_year_0], [birthday_day_1, ...]]
    """
    strip_split_dmy = lambda col: re.sub(r"(?P<dmy>_day|_month|_year)(?P<num>_\d+)?$",r"\g<num>",col)
    col_names = {strip_split_dmy(col) for col in cols }
    col_cluster = [[ col for col in cols if strip_split_dmy(col) == col_name ] for col_name in col_names]
    return col_cluster

def split_cols_by_basename(cols):
    """
        Separates a collection of columns by their base name
        Example: {name_1, name_2, other_name_1} => [[name_1, name_2], [other_name_1]]
    """
    strip_split_number = lambda col: re.sub(r"_\d+$","",col)
    col_names = {strip_split_number(col) for col in cols }
    col_cluster = [sorted([col for col in cols if strip_split_number(col) == col_name ]) for col_name in col_names]
    return col_cluster

def recombine_col_split(col_slice, join_str=" "):
    strip_split_number = lambda col: re.sub(r"_\d+$","",col)
    def reduce_row(row):
        non_na_vals = [ str(v).strip() for v in row if pd.notna(v) ]
        return join_str.join(non_na_vals)
    recombined_columns = col_slice.apply(reduce_row ,axis="columns")
    recombined_columns_df = pd.DataFrame(recombined_columns,columns=[strip_split_number(col_slice.columns[0])])
    return recombined_columns_df

# -----------------------   Strings  ------------------------------------

replacements = {
        'á': 'a',        'ï': 'i',        'ş': 's',        'ó': 'o',
        'ł': 'l',        'ñ': 'n',        'è': 'e',        'ç': 'c',
        'ß': 'ss',        'ô': 'o',        'ü': 'u',        'æ': 'ae',
        'ø': 'o',        'û': 'u',        'ã': 'a',        'ê': 'e',
        'ë': 'e',        'ù': 'u',        'ï': 'i',        'î': 'i',
        'é': 'e',        'í': 'i',        'ú': 'u',        'ý': 'y',
        'à': 'a',        'ì': 'i',        'ò': 'o',        'ã': 'a',
        'ñ': 'n',        'õ': 'o',        'ç': 'c',        'ă': 'a',
        'ā': 'a',        'ē': 'e',        'ī': 'i',        'ō': 'o',
        'ū': 'u',        'ȳ': 'y',        'ǎ': 'a',        'ě': 'e',
        'ǐ': 'i',        'ǒ': 'o',        'ǔ': 'u',        'ǜ': 'u',
        'ǽ': 'ae',        'ð': 'd',        'œ': 'oe',        'ẽ': 'e',
        'ỹ': 'y',        'ũ': 'u',        'ȩ': 'e',        'ȯ': 'o',
        'ḧ': 'h',        'ẅ': 'w',        'ẗ': 't',        'ḋ': 'd',
        'ẍ': 'x',        'ẁ': 'w',        'ẃ': 'w',        'ỳ': 'y',
        'ÿ': 'y',        'ỹ': 'y',        'ŷ': 'y',        'ą': 'a',
        'į': 'i',        'ś': 's',        'ź': 'z',        'ć': 'c',
        'ń': 'n',        'ę': 'e',        'ţ': 't',        'ģ': 'g',
        'ķ': 'k',        'ņ': 'n',        'ļ': 'l',        'ż': 'z',
        'ċ': 'c',        'š': 's',        'ž': 'z',        'ď': 'd',
        'ľ': 'l',        'ř': 'r',        'ǧ': 'g',        'ǳ': 'dz',
        'ǆ': 'dz',        'ǉ': 'lj',        'ǌ': 'nj',        'ǚ': 'u',
        'ǘ': 'u',        'ǜ': 'u',        'ǟ': 'a',        'ǡ': 'a',
        'ǣ': 'ae',        'ǥ': 'g',        'ǭ': 'o',        'ǯ': 'z',
        'ȟ': 'h',        'ȱ': 'o',        'ȹ': 'y',        'ḭ': 'i',
        'ḯ': 'i',        'ḱ': 'k',        "=": "-"
    }

umlaut_replacements = {
    'ä': 'ae',    #'ae': 'a',
    'ö': 'oe',    #'oe': 'a',
    'ü': 'ue',    #'(?<!a)ue': 'a',
    'ß': 'ss',
}

phonetic_replacements = {
    'th': 't',
    'ck': 'k',
    'ph': 'f',
    'w': 'v',
    'y': 'i',
    'j': 'i',
    'tz': 'z',
}

def replace_special_character(name: str):
    for pattern,replace in replacements.items():
        name = re.sub(pattern,replace,name)
    return name

def replace_umlaut_character(name: str):
    for pattern,replace in umlaut_replacements.items():
        name = re.sub(pattern,replace,name)
    return name

def replace_phonetic_character(name: str):
    for pattern,replace in phonetic_replacements.items():
        name = re.sub(pattern,replace,name)
    return name

def remove_double_characters(name:str):
    # Use regex to replace consecutive double characters with single characters
    return re.sub(r'([a-zA-Z])\1', r'\1', name)

def remove_lang_specific_last_name_endings(name):
    name = re.sub(r'owa$|ova$','',name)
    name = re.sub(r'sohns$','sons',name)
    name = re.sub(r'sohn$','son',name)
    name = re.sub(r'(?<=sk|ck)a$','i',name)
    return name