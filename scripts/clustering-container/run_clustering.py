import sys
import pandas as pd
from aroa_etl.attribute_processing.string_utils import preprocess_name, preprocess_last_name
import pickle

fname = sys.argv[1]

# parameters
iteration = "fast"                 # "fast" or None
linkage = "max"                    # max single or avg
cutoff = 90                        # range 0 to 100 
num_perm = 8
lsh_threshold = 0.7
minhash_kwargs = {"num_perm": num_perm,} # permutation of lsh. The higher the num_perm the more likely similar persons are pre-clustered
lsh_kwargs = {
    "threshold":lsh_threshold,     # The degree of pre-cluster similarity. Greater 0 means that person names share some parts.
    "num_perm": num_perm,      # permutation of lsh. The higher the num_perm the more likely similar persons are pre-clustered
    "weights": (0.4,0.6)
} # weight on false positives vs false negatives (precision vs recall) 
leave_one_out_hashing = False
idx_chars = 4 # bucker parameter
len_chars = 2

person_data = pd.read_csv(fname,sep="|")
person_data = person_data.fillna("").astype(str)
#show(person_data)

print("Preprocess data")

person_data["strLName_processed"] = person_data["strLName"].apply(preprocess_last_name)
person_data["strGName_processed"] = person_data["strGName"].apply(preprocess_name)
person_data["strDoB_processed"] = person_data["strDoB"].fillna('00000000')
person_data["strPoB_processed"] = person_data["strPoB"].apply(preprocess_name)
person_data["prisoner_number"] = person_data["prisoner_number"].astype(str)
person_data["prisoner_number"] = person_data["prisoner_number"].mask(person_data["prisoner_number"] == "0").mask(person_data["prisoner_number"] == "-1" )

print("Group multiple names")
agg_person_data = person_data.groupby(['lObjId', 'lCountId']).apply(lambda group: [" ".join(group["strGName_processed"].values),
                                                                 " ".join(group["strLName_processed"].values),
                                                                 group["strDoB_processed"].iloc[0],
                                                                 group["strPoB_processed"].iloc[0],
                                                                 group["prisoner_number"].iloc[0],
                                                                 group["TD_number"].iloc[0]] )

person_data = pd.DataFrame(list(agg_person_data),columns=["strGName_processed","strLName_processed","strDoB_processed","strPoB_processed","prisoner_number","TD_number"],index = agg_person_data.index)
person_data = person_data.reset_index()
from aroa_etl.person_matching.person_clustering import build_buckets, get_buckets_for_name

print("start building buckets")
# lsh, minhashes = local_semantic_hashing(person_data,minhash_kwargs,lsh_kwargs,leave_one_out_hashing)
# get_bucket_fn = lambda person_idx: lsh.query(minhashes[person_idx])

first_name_buckets = build_buckets(person_data, column="strGName_processed", idx_chars=idx_chars,len_chars=len_chars)
last_name_buckets = build_buckets(person_data, column="strLName_processed", idx_chars=idx_chars,len_chars=len_chars)

def build_get_bucket_fn():
    _person_data = person_data.copy()
    def get_bucket_fn(person_idx):
        fname_buckets = get_buckets_for_name(_person_data["strGName_processed"][person_idx],idx_chars)
        lname_buckets = get_buckets_for_name(_person_data["strLName_processed"][person_idx],idx_chars)
        first_bucket = set()
        for fbucket in fname_buckets:
            first_bucket = first_bucket.union(first_name_buckets[fbucket])
        last_bucket = set()
        for lbucket in lname_buckets:
            last_bucket = last_bucket.union(last_name_buckets[lbucket])
        return first_bucket.intersection(last_bucket)
    return get_bucket_fn


#breakpoint()
print("Compute known Clusters")
td_cases = person_data["TD_number"][person_data["TD_number"]!= ''].str.replace(".0","")
td_clusters = td_cases.to_frame().groupby("TD_number")
td_clusters = [list(group.index) for _, group in td_clusters]
td_cluster_map = {idx : cluster for cluster in td_clusters for idx in cluster}

from aroa_etl.person_matching.person_clustering import agglomerative_clustering, cluster_column, clean_td_cases
print("start clustering")

clustering = agglomerative_clustering(build_get_bucket_fn(), td_cluster_map, person_data, cutoff, linkage, iteration)

print("Add Person Entity ID")
person_data["Person_Entity_ID"] = cluster_column(person_data, clustering)

outname = fname.split(".")
outname = f"{outname[0]}_with_clusters_{iteration}_{linkage}_cutoff_{cutoff}"
person_data.to_csv(f"{outname}.csv")
person_data.to_pickle(f"{outname}.pkl")

# print("clean TD Cases")
# person_data = clean_td_cases(person_data)

# print("Dump result")
# #with open(f'clustering_{linkage}_cutoff_{cutoff}_num_perm_{num_perm}_lsh_cutoff_{lsh_threshold}.pkl', 'wb') as file:
# #    pickle.dump(clustering, file) 
# person_data.to_csv(f"{outname}.csv")
# person_data.to_pickle(f"{outname}.pkl")

print(f"finished clustering")
breakpoint()
