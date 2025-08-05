import pandas as pd
from pandas.core.frame import DataFrame
from aroa_etl.attribute_processing.string_utils import preprocess_name, preprocess_last_name
from aroa_etl.person_matching.matching import person_matching
import sys
from aroa_etl.utils import value_is_not_empty_q
import os
from tqdm import tqdm
import pickle

external_fname = sys.argv[1]

def preprocess_persdata(persdata: DataFrame):
    print("preprocess persdata")
    persdata["strLName_processed"] = persdata["strLName"].fillna("").apply(preprocess_last_name)
    persdata["strGName_processed"] = persdata["strGName"].fillna("").apply(preprocess_name)
    persdata["strDoB_processed"] = persdata["strDoB"].fillna('00000000').astype(str).str.replace(r'^[^\d]*$', '00000000', regex=True)
    persdata["prisoner_number"] = persdata["strPrisNo"]
    #persdata = persdata[["lObjId", "lCountId","strGName","strLName", "strGName_processed","strLName_processed","strDoB_processed","prisoner_number", "TDNumber"]]
    agg_functions = {
        "strGName_processed": lambda values: " ".join(list(v for v in set(values) if value_is_not_empty_q(v))),
        "strLName_processed": lambda values: " ".join(list(v for v in set(values) if value_is_not_empty_q(v))),
        "strDoB_processed" : lambda values: values.iloc[0],
        "prisoner_number": lambda values: values.iloc[0], 
        "TDNumber" : lambda values: values.iloc[0]
    }
    #persdata = persdata.groupby(["lObjId", "lCountId"]).agg(agg_functions)
    #persdata.reset_index()
    persdata.to_pickle("/persdata/persdata_processed.pkl")
    return persdata

print("load external data")
external = pd.read_csv(external_fname, sep='|') if external_fname[-3:] == "csv" else pd.read_excel(external_fname)
print("preprocess external")
external["geburt_jahr"] = external["geburt_jahr"].astype(str).fillna("0000").str.replace(r"^0$","0000",regex=True)
external["geburt_monat"] = external["geburt_monat"].astype(str).fillna("00").str.replace(r"^0$","00",regex=True).apply(lambda m: "0"*(2-len(m)) + m)
external["geburt_tag"] = external["geburt_tag"].astype(str).fillna("00").str.replace(r"^0$","00",regex=True).apply(lambda d: "0"*(2-len(d)) + d)
external["strDoB_processed"] = external["geburt_jahr"] + external["geburt_monat"] + external["geburt_tag"]
external["strLName_processed"] = external["nachname"].fillna("").apply(preprocess_last_name)
external["strGName_processed"] = external["vorname"].fillna("").apply(preprocess_name)
#external["DateOfBirth"] = external["DateOfBirth"].astype(str).str.replace(r'^[^\d]*$', '00000000', regex=True)
#external["strDoB_processed"] = external["DateOfBirth"].fillna('00000000')

print("compute matchings")
matchings = []
for batch_file in tqdm(os.listdir("/persdata/query_batches/")):
    with open(f"/persdata/query_batches/{batch_file}", "rb") as f: 
        batch = pickle.load(f)
    persdata_batch = pd.DataFrame(batch)
    persdata_batch = preprocess_persdata(persdata_batch)
    matchings_df = person_matching(external, persdata_batch, allow_duplicates=True,
                    src_gname_col="strGName_processed", target_gname_col="strGName_processed",
                    src_lname_col="strLName_processed", target_lname_col="strLName_processed",
                    src_date_col="strDoB_processed", target_date_col="strDoB_processed",
                    trg_pre_clustering_on_n_chars=2, trg_pre_clustering_group_n_len_units=4, 
                    top_n_matches = 10, min_match_score=80.0, name_only=False)
    # cleanup match columns
    persdata_batch = persdata_batch.drop(["strGName_processed", "strLName_processed"],axis=1)
    # merge matching indices
    external_matched = pd.merge(external, matchings_df, left_index=True, right_on='srcID')
    external_matched = pd.merge(external_matched, persdata_batch, how="left", left_on="trgID",right_index=True,suffixes=("","_AroA"))
    external_matched.drop(["srcID","trgID","strGName_processed", "strLName_processed"], inplace=True, axis="columns")
    external_matched = external_matched.fillna("")
    matchings.append(external_matched)

matchings_df = pd.concat(matchings)

matchings = []
for v in set(matchings_df["id"].values):
    matching_slice = matchings_df.loc[(matchings_df["id"] == v),:]
    non_matches = matching_slice.loc[(matching_slice.score == -1), :]
    matches = matching_slice.loc[(matching_slice.score != -1), :]
    if matches.shape[0] >0:
        matchings.append(matches)
    else:
        matchings.append(non_matches.iloc[0,:])

external_matched = pd.concat(matchings,ignore_idex=True).sort_values(by="id")
external_matched.index = range(external_matched.shape[0])
external_matched = external_matched[['id', 'nachname', 'vorname', 'geburt_jahr', 'geburt_monat', 'geburt_tag', 'score', 'strSchemaCode', 'lObjId', 'lCountId', 'strLName', 'strGName', 'strDoB', 'prisoner_number']]

external_matched.rename({
    "id": "Fremdsignatur",
    "nachname": "Nachname",
    "vorname": "Vorname",
    "score": "Match Score",
    'strSchemaCode' : "Bestand",
    'lObjId' : "Objekt ID",
    'lCountId': "Count ID",
    'strLName' : "Nachname AroA",
    'strGName': "Vorname AroA",
    'strDoB': "Geburtsdatum",
    'prisoner_number': "Häftlingsnummer"
}, axis=1, inplace=True)

print(f"save results")
outfname = external_fname.split(".")[0]
external_matched.to_pickle(f"{outfname}_matched.pkl")
external_matched.to_csv(f"{outfname}_matched.csv", index=False, sep="|")
external_matched.to_excel(f"{outfname}_matched.xlsx")
breakpoint()
