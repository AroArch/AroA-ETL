from  rapidfuzz import fuzz, utils
import re
import numpy as np
import math
import pandas as pd
from aroa_etl.attribute_processing.string_utils import preprocess_name, preprocess_last_name
from tqdm import tqdm
from rapidfuzz import fuzz, utils

# ------------------------- Person Similarity Measure ---------------------------------

def number_diff(num_1: int, num_2: int):
    difference =  (5 ** abs(num_1 - num_2)) - 1
    score = max(0, 100 - difference)
    return score

def day_month_score(day_1:int,day_2:int,month_1:int,month_2:int):
    # compute day and month score
    month_score = number_diff(month_1,month_2)
    month_score = -1 if month_1 ==0 or month_2 ==0 else month_score
    
    day_score = number_diff(day_1,day_2)
    day_score = -1 if day_1 == 0 or day_2 == 0 else day_score
    return month_score, day_score

def compute_year_score(year_1:int,year_2:int):
    year_score = number_diff(year_1, year_2)
    year_score = -1 if year_1 == 0 or year_2 == 0 else year_score    
    return year_score

def parse_date(date: str) -> tuple[int,int,int]|None:
    parsed_date = None
    # yyyymmdd(.0)
    parsed_date = re.match(r"^(?P<year>\d\d\d\d)(?P<month>\d\d)(?P<day>\d\d)\.?0?$",date)
    if not parsed_date is None:
        year,month,day = map(int, parsed_date.groups())
        return year,month,day
    parsed_date = re.match(r"^(?P<day>\d\d)\.(?P<month>\d\d)\.(?P<year>\d\d\d\d)$",date)
    if not parsed_date is None:
        day, month, year = map(int, parsed_date.groups())
        return year,month,day
    return None
        

def date_similarity(date_1:str,date_2:str):
    """
        More complex similarity measure for dates
    """
    date_1 = parse_date(str(date_1))
    date_2 = parse_date(str(date_2))

    if date_1 is None or date_2 is None:
        return -1
    
    year_1,month_1,day_1 = date_1
    year_2,month_2,day_2 = date_2
        
    year_score = compute_year_score(year_1,year_2)

    # compute day and month score
    month_score, day_score = day_month_score(day_1,day_2,month_1,month_2)
    
    # check reversed 
    month_score_reversed, day_score_reversed = day_month_score(day_1,month_2,month_1,day_2)

    if month_score + day_score <= month_score_reversed + day_score_reversed:
        month_score, day_score = month_score_reversed, day_score_reversed

    score_list = [year_score, month_score, day_score]
    score = 100
    for s in score_list:
        if s>=0:
            score = score - (100-s)
    return -1 if len(score_list) == 0 else max(0, score)

def __not_empty(field):
    return pd.notna(field) and len(field)>0 and "".join(field)!="" and field != "00000000" and field != "-1.0" and field != "-1"

def simple_date_matcher(src_date: str, target_date: str):
    """
        Fuzzy matching for dates in dd.mm.yyyy format 
    """
    score = -1
    if __not_empty(src_date) and __not_empty(target_date):
        src_date_parts = re.findall(r"[1-9]\d*",src_date)
        trg_date_parts = re.findall(r"[1-9]\d*",target_date)
        score = min(3,len([1 for date_part in src_date_parts if date_part in trg_date_parts]))/3
        score = score * 100
    return score
    

def name_matcher(src_name: str, target_name: str):
    """
        Fuzzy matching for names. Two empty/nan names are treated as match.
    """
    score = -1
    if __not_empty(src_name) and __not_empty(target_name):
        score = fuzz.ratio(src_name,target_name,processor=utils.default_process)
        score = score
    return score

def name_set_matcher(src_name: str, target_name: str):
    """
        Fuzzy matching for names. Two empty/nan names are treated as match. Order of names is ignored
    """
    score = -1
    if __not_empty(src_name) and __not_empty(target_name):
        score = fuzz.token_set_ratio(src_name,target_name,processor=utils.default_process)
        score = score
    return score


def person_similarity(src_person: pd.core.series.Series, trg_person: pd.core.series.Series,
                      src_gname_col="strGName_processed",src_lname_col="strLName_processed",src_date_col="strDoB_processed",
                      src_prisoner_number="prisoner_number",src_birthplace = "strPoB_processed",
                      target_gname_col="strGName_processed",target_lname_col="strLName_processed",target_date_col="strDoB_processed",
                      target_prisoner_number="prisoner_number",target_birthplace = "strPoB_processed",
                      date_matcher=date_similarity, name_only=False, non_names_optional=False
                      ):
    # primary
    primary_scores = []
    if src_lname_col in src_person:
        score = max(0,name_set_matcher(src_person[src_lname_col], trg_person[target_lname_col]))
        primary_scores.append(score)
    if src_gname_col in src_person:
        score = max(0,name_set_matcher(src_person[src_gname_col], trg_person[target_gname_col]))
        primary_scores.append(score)
    primary_scores = [s for s in primary_scores if s>=0]
    primary_score = np.sum(primary_scores)/2 if len(primary_scores) > 0 else 0
    if name_only:
        return primary_score 
    # secondary ids
    secundary_scores = []
    if src_prisoner_number in src_person:
        score = name_matcher(src_person[src_prisoner_number], trg_person[target_prisoner_number])
        secundary_scores.append(score)
    if src_date_col in src_person:
        score = max(0,date_matcher(src_person[src_date_col],trg_person[target_date_col]))
        secundary_scores.append(score)
    secundary_scores = [s for s in secundary_scores if s>=0]
    if len(secundary_scores) > 0:
        secundary_score = np.array(secundary_scores).mean()
    elif non_names_optional:
        secundary_score = -1
    else:
        secundary_score= 0
    # other
    other_scores = []
    if src_birthplace in src_person:
        score = name_matcher(src_person[src_birthplace],trg_person[target_birthplace])
        other_scores.append(score)
    other_scores = [s for s in other_scores if s>=0]
    if len(other_scores) > 0:
        other_score = np.array(other_scores).mean()
    else:
        other_score = -1

    # combine with weights
    score = primary_score
    if secundary_score >= 0:
        score = 2/3 * score + 1/3 * secundary_score
    if other_score >=0:
        score =  3/4 * score + 1/4 * other_score
    return score 
