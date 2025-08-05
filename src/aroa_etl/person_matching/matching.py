import pandas as pd
import numpy as np
from iteration_utilities import first
from rapidfuzz import process, fuzz, utils
import re 
from collections import defaultdict
from tqdm import tqdm
from aroa_etl.person_matching.similarity_measures import simple_date_matcher, date_similarity, person_similarity, name_matcher
    
def compute_trg_buckets(target_df, target_gname_col, target_lname_col, trg_pre_clustering_on_n_chars=2, trg_pre_clustering_group_n_len_units=4):
    target_fname_buckets = defaultdict(list)
    target_lname_buckets = defaultdict(list)

    get_key = lambda name: get_bucket_key(name, trg_pre_clustering_on_n_chars, trg_pre_clustering_group_n_len_units)

    for idx, row in tqdm(target_df.iterrows(),total=target_df.shape[0]):
        fname = row[target_gname_col]
        for subname in re.sub(r"[^a-z\s]","",fname).split(" "):
            target_fname_buckets[get_key(subname)].append(idx)
        lname = row[target_lname_col]
        for subname in re.sub(r"[^a-z\s]","",lname).split(" "):
            target_lname_buckets[get_key(subname)].append(idx)
    return target_fname_buckets, target_lname_buckets
        
def get_bucket_key(name, trg_pre_clustering_on_n_chars, trg_pre_clustering_group_n_len_units):
    return (name[:trg_pre_clustering_on_n_chars],int(len(name)/trg_pre_clustering_group_n_len_units))


def person_matching(src_df, target_df, allow_duplicates=True,
                    src_gname_col="strGName_processed",src_lname_col="strLName_processed",src_date_col="strDoB_processed",
                    src_prisoner_number="prisoner_number",src_birthplace = "strPoB_processed",
                    target_gname_col="strGName_processed",target_lname_col="strLName_processed",target_date_col="strDoB_processed",
                    target_prisoner_number="prisoner_number",target_birthplace = "strPoB_processed", date_matcher=date_similarity, 
                    trg_pre_clustering_on_n_chars=2, trg_pre_clustering_group_n_len_units=4, 
                    top_n_matches = 1, min_match_score=0.0, name_only=False):
    """
        Computes a matching between documents in `src_df` and documents in `target_df` based on person data. 
        The documents are fuzzy matched with threshold `matching_threshold`. Excluding duplicates from two 
        src_docs to the same target is not yet implemented.
    """
    matching = []
    print("Precluster target dataframe ")
    target_fname_buckets, target_lname_buckets = compute_trg_buckets(
        target_df,
        target_gname_col,
        target_lname_col, 
        trg_pre_clustering_on_n_chars, 
        trg_pre_clustering_group_n_len_units
    )
    get_key = lambda name: get_bucket_key(name, trg_pre_clustering_on_n_chars, trg_pre_clustering_group_n_len_units)
    print("Start Matching ")
    for src_idx, src_doc in tqdm(src_df.iterrows(), total = src_df.shape[0]):
        best_matches = [] # list of (score, idx) in increasing order
        # get target candidates for matching
        fname = src_doc[src_gname_col]
        fname = re.sub(r"[^a-z\s]","",fname)
        fname_bucket = [idx for subname in fname.split(" ") for idx in target_fname_buckets[get_key(subname)]]
        lname = src_doc[src_lname_col]
        lname = re.sub(r"[^a-z\s]","",lname)
        lname_bucket = [idx for subname in lname.split(" ") for idx in target_lname_buckets[get_key(subname)]]
        bucket_idxs = list(set(fname_bucket).intersection(set(lname_bucket)))
        #print(num_match_columns)
        for target_idx,target_doc in target_df.iloc[bucket_idxs,:].iterrows():
            match_score = person_similarity(
                src_doc, target_doc,
                src_gname_col=src_gname_col,src_lname_col=src_lname_col,src_date_col=src_date_col,
                target_gname_col=target_gname_col,target_lname_col=target_lname_col,target_date_col=target_date_col, date_matcher=date_matcher,
                name_only=name_only
            )
            ranking_pos = -1
            if match_score >= min_match_score:
                ranking_pos = 0                
            for top_score, idx in best_matches:
                if match_score > top_score:
                    ranking_pos += 1
            if ranking_pos >=0:
                best_matches = best_matches[:ranking_pos] + [(match_score, target_idx)] + best_matches[ranking_pos:]
            if len(best_matches) > top_n_matches:
                best_matches = best_matches[1:]
        if len(best_matches) == 0:
            best_matches = [(-1, np.nan)]
        best_matches = [ (src_idx, match_score, match_idx) for match_score, match_idx in best_matches ]
        # best_match_idx is -1 if there is no match score greater than matching_threshold
        matching += best_matches
    matchings_df = pd.DataFrame(matching,columns=["srcID","score","trgID"])
    if not allow_duplicates:
        best_matches_per_trg = matchings_df[matchings_df.score != -1.0].groupby("trgID")["score"].agg("max").reset_index()
        _matchings_df = pd.merge(matchings_df, best_matches_per_trg, how="right", left_on=["score","trgID"], right_on=["score","trgID"])
        _matchings_df = _matchings_df.drop_duplicates(["score", "trgID"])
        non_matched_srcids = matchings_df.srcID[~matchings_df.srcID.isin(set(_matchings_df.srcID.values))]
        non_matched_srcids = pd.DataFrame([[srcid, -1, pd.NA] for srcid in non_matched_srcids], columns=["srcID","score","trgID"])
        matchings_df = pd.concat([_matchings_df, non_matched_srcids], axis=0)
        matchings_df.index = range(matchings_df.shape[0])
    return matchings_df
