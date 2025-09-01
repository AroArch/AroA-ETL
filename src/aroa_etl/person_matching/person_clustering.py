from  rapidfuzz import fuzz, utils
import re
import numpy as np
from datasketch import MinHash, MinHashLSH
import math
import pandas as pd
from aroa_etl.attribute_processing.string_utils import preprocess_name, preprocess_last_name
from aroa_etl.person_matching.similarity_measures import *
from tqdm import tqdm
from collections import defaultdict  
from typing import Dict

# ------------------------- Cluster Quality measures ---------------------------------


def avg_link_score(person:pd.core.series.Series,
                   person_cluster:"list | pd.core.frame.DataFrame"):
    if type(person_cluster) == list and len(person_cluster) == 0:
        return 100
    elif type(person_cluster) == pd.core.frame.DataFrame and person_cluster.shape[0] == 0:
        return 100

    score = 0 
    if type(person_cluster) == list:
        score = np.array([person_similarity(person,cluster_person) for cluster_person in person_cluster]).mean()
    elif type(person_cluster) == pd.core.frame.DataFrame:
        score = np.array([person_similarity(person,cluster_person) for _, cluster_person in person_cluster.iterrows()]).mean()
    return score

def single_link_score(person:pd.core.series.Series,
                   person_cluster:"list | pd.core.frame.DataFrame"):
    if type(person_cluster) == list and len(person_cluster) == 0:
        return 100
    elif type(person_cluster) == pd.core.frame.DataFrame and person_cluster.shape[0] == 0:
        return 100

    score = 0 
    if type(person_cluster) == list:
        score = np.array([person_similarity(person,cluster_person) for cluster_person in person_cluster]).max()
    elif type(person_cluster) == pd.core.frame.DataFrame:
        score = np.array([person_similarity(person,cluster_person) for _, cluster_person in person_cluster.iterrows()]).max()
    return score

def max_link_score(person:pd.core.series.Series,
                   person_cluster:"list | pd.core.frame.DataFrame"):
    if type(person_cluster) == list and len(person_cluster) == 0:
        return 100
    elif type(person_cluster) == pd.core.frame.DataFrame and person_cluster.shape[0] == 0:
        return 100

    score = 0 
    if type(person_cluster) == list:
        score = np.array([person_similarity(person,cluster_person) for cluster_person in person_cluster]).min()
    elif type(person_cluster) == pd.core.frame.DataFrame:
        score = np.array([person_similarity(person,cluster_person) for _, cluster_person in person_cluster.iterrows()]).min()
    return score

def link_score(person:pd.core.series.Series,
               person_cluster:"list | pd.core.frame.DataFrame",
               linkage: str):
    if linkage == "single":
        return single_link_score(person,person_cluster)
    elif linkage == "average":
        return avg_link_score(person,person_cluster)
    elif linkage == "max":
        return max_link_score(person,person_cluster)
    assert False, "Linkage not defined"

def cluster_integrety(person_cluster:"list | pandas.core.frame.DataFrame"):
    if type(person_cluster) == pd.core.frame.DataFrame:
        person_cluster = [person for i,person in person_cluster.iterrows()]
    avg_link_scores = np.array([avg_link_score(person,[*person_cluster[:i],*person_cluster[i+1:]])
                                for i,person in enumerate(person_cluster) ])
    single_link_scores = np.array([single_link_score(person,[*person_cluster[:i],*person_cluster[i+1:]])
                                   for i,person in enumerate(person_cluster) ])
    
    avg_link_scores = np.array([100]) if len(person_cluster) == 0 else avg_link_scores
    single_link_scores = np.array([100]) if len(person_cluster) == 0 else single_link_scores
    return {"average-link": avg_link_scores.mean(), "single-link" : single_link_scores.mean()}

def jaccard_distance_cluster(cl1,cl2):
    cl1 = set(cl1)
    cl2 = set(cl2)
    return len(cl1.intersection(cl2))/len(cl1.union(cl2))

# --------------------------- Build buckets for pre-clustering ---------------------------------

def add_windowed_collision_hashes(minhash, name):
    """
        Add alterations of name to the hash object. This is done to provoce collisions with similar persons.
        The alterations are windows within the `name` string. This also allows for substring collisions.
        Hashes are computed on non-vocal characters only.
    """
    # reduce to non-vocal
    name = re.sub(r"[aeiou]", "", name)
    # add hash of full name
    minhash.update(name.encode('utf8'))
    # define window size
    character_cnt = len(name)
    window_size = math.ceil(character_cnt / 2)
    window_fits_cnt = character_cnt + 1 - window_size
    for c in range(window_fits_cnt):
        # add hash of window
        name_to_hash = name[c:c+window_size].encode('utf8')
        minhash.update(name_to_hash)
        
def add_collision_hashes(minhash,name,remove_vocals=False):
    """
        Add alterations of name to the hash object. This is done to provoce collisions with similar persons.
        The alterations omit one character of `name` to account for misspelling. Hashes are computed on non-vocal characters only.
    """
    # reduce to non-vocal
    if remove_vocals:
        name = re.sub(r"[aeiou]","",name)
    # add hash of full name
    minhash.update(name.encode('utf8'))
    for c in range(len(name)):
        # add hash of misspelled by one character
        name_to_hash = "".join([*name[:c],*name[c+1:]]).encode('utf8')
        minhash.update(name_to_hash)

def local_semantic_hashing(person_data: pd.core.frame.DataFrame,
                           minhash_kwargs: dict  = {"num_perm": 128,},
                           lsh_kwargs :dict  = {"threshold":0.01, 
                                               "num_perm": 128, 
                                               "weights": (0.4,0.6)},
                          leave_one_out_hashing : bool = False):
    """
        Compute a local semantic hashing for person information in `person_data`.
        Returns an indexing object to find persons that are similarly hashed and a map for every persons hash.
        The hashing is computed on non vocal characters only to provoce collisions with similar persons.
    """
    # hash db
    lsh = MinHashLSH(**lsh_kwargs)
    # hashes of every person
    minhashes = dict()
    for i, person in tqdm(person_data.iterrows(),total=person_data.shape[0]):
        # a persons hashes
        minhash = MinHash(**minhash_kwargs)
        # include last name in hashing
        for sub_name in person["strLName_processed"].split(" "):
            minhash.update(sub_name.encode('utf8'))
            if leave_one_out_hashing:
                add_collision_hashes(minhash,sub_name)
        # include first name in hashing
        for sub_name in person["strGName_processed"].split(" "):
            minhash.update(sub_name.encode('utf8'))
            if leave_one_out_hashing:
                add_collision_hashes(minhash,sub_name)
            #minhash.update(sub_name.encode('utf8'))
        lsh.insert(i,minhash)
        minhashes[i] = minhash
    return lsh, minhashes

def get_buckets_for_name(name, idx_chars,len_chars=3):
    return [(sub_name.lower()[:idx_chars], int(len(sub_name)/len_chars)) for sub_name in name.split(" ") if len(sub_name)>=idx_chars]

def build_buckets(person_data, column, idx_chars=3,len_chars=3):
    bucket = defaultdict(set)
    print(f"build buckets for {column}")
    for idx, field in tqdm(person_data[column].items(),total=person_data.shape[0]):
        for bucket_name in get_buckets_for_name(field, idx_chars):
            bucket[bucket_name].add(idx)
    return bucket


# ------------------- Clustering ---------------------------------------------

def local_agglomerative_cluster_fast(pre_cluster:list[int], person_bucket: pd.core.frame.DataFrame,cutoff: float,linkage: str, link_cascade=False):
    """
        Uses agglomerative clustering to compute the cluster of an idividual i. This faster version does only compute linkage score to the initial person i
        Returns an index list
    """
    person_cluster = pre_cluster
    # persons that do not (yet) belong to a the person_cluster
    other_persons_idxs = person_bucket.index.difference(person_cluster)
    for other_person_idx, other_person in person_bucket.loc[other_persons_idxs,:].iterrows():
        # the score on how likely other_person belongs to the same person
        score = link_score(other_person,person_bucket.loc[person_cluster,:],linkage)
        if score >=cutoff:
            # add other person to the cluster
            person_cluster.append(other_person_idx)
    return person_cluster

def local_agglomerative_cluster(pre_cluster:list[int], person_bucket: pd.core.frame.DataFrame, cutoff: float,linkage: str, iteration:str, link_cascade=False, ):
    """
        Uses agglomerative clustering to compute the cluster of an idividual i.
        Returns an index list
    """
    if iteration == "fast":
        return local_agglomerative_cluster_fast(
            pre_cluster = pre_cluster,
            person_bucket = person_bucket,
            cutoff = cutoff,
            linkage = linkage,
            link_cascade=link_cascade
        )
    person_cluster = pre_cluster
    # check if an iteration found no new additions
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        # persons that do not (yet) belong to a the person_cluster
        other_persons_idx = person_bucket.index.difference(person_cluster)
        for other_person_idx, other_person in person_bucket.loc[other_persons_idx,:].iterrows():
            # the score on how likely other_person belongs to the same person
            score = link_score(other_person,person_bucket.loc[person_cluster],linkage)
            if score >=cutoff:
                # add other person to the cluster
                person_cluster.append(other_person_idx)
                cluster_changed = link_cascade and True
    return person_cluster

def preprocess_clustering_data(
        person_data: pd.core.frame.DataFrame,
        gname_col="strGName_processed", lname_col="strLName_processed"
) -> pd.core.frame.DataFrame:
    person_data[gname_col] = person_data[gname_col].apply(preprocess_name)
    person_data[lname_col] = person_data[lname_col].apply(preprocess_last_name)
    return person_data

def agglomerative_clustering(get_bucket_fn,
                             known_clusters: Dict[int, list[int]],
                             person_data: pd.core.frame.DataFrame,
                             cutoff: float,
                             linkage: str,
                             iteration: str,
                             allow_known_cluster_merge = False
                             ):
    """
        This method computes an agglomerative clustering on person_data to build persons.
        Returns a list of index lists.
        This method assumes that no pre-known clusters are merged. They can be extended.
    """
    not_clustered = person_data.index
    # enumerate known clusters first.
    enumeration_order = list(known_clusters.keys()) + list(not_clustered.difference(known_clusters.keys()))
    not_clustered = not_clustered[enumeration_order]
    clustering = []
    num_person_rows = not_clustered.shape[0]
    known_cluster_map = lambda idx: known_clusters[idx] if idx in known_clusters else [idx]
    pre_clustered_entities = set(known_clusters.keys())
    print(f"{num_person_rows} Person Rows")
    print(f"Preprocess Person Data")
    _person_data = preprocess_clustering_data(person_data)
    with tqdm(total=num_person_rows) as pbar:
        while len(not_clustered) > 0:
            person_idx = not_clustered[0]
            # known pre-clustering for td cases or other sources
            pre_cluster = known_cluster_map(person_idx)
            # get similar persons from the lsh index
            # person_bucket = lsh.query(minhashes[person_idx])
            person_bucket = {bucket_idx for idx in pre_cluster for bucket_idx in get_bucket_fn(idx)}
            if not allow_known_cluster_merge:
                person_bucket = person_bucket.difference(pre_clustered_entities)
            person_bucket = list(person_bucket)
            # reduce to person_information that are not already clustered
            person_bucket = not_clustered.intersection(person_bucket)
            person_bucket = person_bucket.union(pre_cluster)
            # slice in which the person_cluster of person_idx is computed 
            person_bucket = _person_data.loc[person_bucket]
            # new person
            person_cluster = local_agglomerative_cluster(pre_cluster,
                                                         person_bucket,
                                                         cutoff,
                                                         linkage,
                                                         iteration,)
            if len(person_cluster) == 0:
                person_cluster = pre_cluster
            clustering.append(person_cluster)
            not_clustered = not_clustered.difference(person_cluster)
            # update 
            pbar.update(len(person_cluster))
    return clustering

# ---------------------- Export  ----------------------------

def cluster_column(person_data, clustering):
    person_cluster_map = dict()
    for cluster_idx, cluster in enumerate(clustering):
        for person in cluster:
            person_cluster_map[person] = cluster_idx
    cluster_col = person_data.index.to_series().apply(lambda idx: int(person_cluster_map[idx])
                                                      if idx in person_cluster_map 
                                                      else None)
    return cluster_col

# ---------------------- Fix Known Clusters  ----------------------------

def clean_td_cases(person_data, td_col="TD_number"):
    td_clustering = person_data[person_data["TD_number"].notna()].groupby(["TD_number"])[["lObjId","lCountId"]].apply(lambda group: list(group.groupby(["lObjId","lCountId"]).groups.keys()))
    clustering = person_data.groupby(["Person_Entity_ID"])[["lObjId","lCountId"]].apply(lambda group: list(group.groupby(["lObjId","lCountId"]).groups.keys()))
    has_known_entity = {idx for cluster in td_clustering for idx in cluster}
    clustering = [[idx for idx in cluster if idx not in has_known_entity] for cluster in clustering]
    clustering = [cluster for cluster in clustering if len(cluster)>0]
    clustering = [*clustering, *list(td_clustering.values)]
    ids_to_row_idx = dict()
    for row_idx, row in person_data.iterrows():
        ids_to_row_idx[(row["lObjId"],row["lCountId"])] = row_idx
    row_idx_clustering = [[ids_to_row_idx[idx] for idx in cluster] for cluster in clustering]
    person_data["Person_Entity_ID"] = cluster_column(person_data, row_idx_clustering)
    return person_data
