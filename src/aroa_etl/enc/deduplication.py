import pandas as pd
from .matching import Enc_Matcher, Col_Matcher, Default_Col_Matcher, Default_Person_Col_Matcher, Default_Date_Col_Matcher, Default_Strict_Col_Matcher, Default_Fuzzy_Col_Matcher
from ..utils import value_is_not_empty_q
import re
import uuid
import numpy as np

class ENC_Deduplicater():
    """
        ENC_Deduplicator objects are used to define deduplication jobs for enc transcriptions.
        The objective is for a DataFrame containing multiple transcriptions of the same document
        to reduce them to one entry per Column. This is done using the aroa_etl.matching.ENC_Matcher 
        Object. The deduplication returns a value based on majority vote. In case there are no two compatible 
        entries per column a np.nan value is returned. The matching strategy can be defined by  
        aroa_etl.matching.Col_Matcher objects. Defaults are availbale for person and date field. 

        id_col | first_name | last_name 
        -------+------------+------------
             1 |        Bob | Mustermann
             1 |        Bob | Mutermann
             1 |        Bod | Mustermann
             2 |      Alice | Nachname   

        For later QA inspections, a flagging routine is implemented with QA Flags for ambiguities found during 
        preprocessing and an ambiguous flagging for matching ambiguities.

        Example usage:
        >>> deduplicater = ENC_Deduplicater(processed_df,"subject_ids")
        >>> deduplicater.on_person_cols(person_cols)
        >>> deduplicater.on_date_cols(date_cols)
        >>> deduplicater.on_other_cols(other_cols)  
        >>> deduplicater.run()

        For debugging of the matching routing check out the aroa_etl.matching.ENC_Matcher documentation.
    """
    
    def __init__(self, data, id_col, metadata_columns=None):
        self.data = data
        self.matcher = Enc_Matcher(data,id_col)
        self.id_col = id_col
        self.person_cols = set()
        self.date_cols = set()
        self.other_cols = set()
        self.other_strict_cols = set()
        self.fuzzy_cols = set()
        self.qa_map = dict()
        self.every_col_has_qa = True
        self.metadata_columns = metadata_columns

    def convert_qa_columns_to_bool(d):
        """
        Function that converts all qa columns into boolean columns
        :param d:
        :return:
        """
        def convert(c):
            if c.dtype == float:
                return c.astype(bool)
            if c.dtype != bool:
                bool_c = c.apply(lambda s: True if s.lower() == 'true' else False)
                return bool_c
            else:
                return c
                
        return d.apply(lambda c: convert(c))

    def deduplication_preprocess(self):
        """
            This function is called before the deduplication routine. Id does:
            - Set default values for missing entries
            - Set QA flags on the row level 
        """
        self.data['has_qa'] = self.data[list(self.qa_map.values())].any(axis=1) 
        
        self.data = self.data.fillna('-').astype(str)

        year_cols = [col for col in self.date_cols if re.search(r"[yY][eE][aA][rR]",col)]
        other_date_cols = [col for col in self.date_cols if col not in year_cols]
        for year_col in year_cols:
            self.data.loc[self.data[year_col]=='-',year_col] = "0000"
        for date_col in other_date_cols:
            self.data.loc[self.data[date_col]=='-',date_col] = "00"

        self.matcher.enc_data = self.data
        
    def deduplication_postprocess(self):
        """
            This function is called after the deduplication routine. Id does:
            - Convert QA columns to boolean
        """
        qa_columns = list(set(self.qa_map.values())) + ["has_qa"]
        self.data[qa_columns] = ENC_Deduplicater.convert_qa_columns_to_bool(self.data[qa_columns])
        
    def assign_qa_flags_to_matching_result(self, match_result):
        """
            Assigns a qa flag to the matching result iff its value is equal flagged value in the processed data set.
        """
        for qa_col in self.qa_map.values():
            match_result[qa_col] = False
        data_with_match_results = pd.merge(self.data, match_result, how="inner", left_on=self.id_col, right_on=self.id_col, suffixes=("","_result"))

        for match_col, qa_col in self.qa_map.items():
            data_with_match_results[f"{match_col}_deduplication_qa"] = (data_with_match_results[match_col] == data_with_match_results[f"{match_col}_result"]) & data_with_match_results[qa_col]
        data_with_match_results_groups = data_with_match_results.groupby(self.id_col)
        for match_col, qa_col in self.qa_map.items():
            match_result_qa = data_with_match_results_groups[f"{match_col}_deduplication_qa"].agg('any')
            match_result[qa_col] = match_result[qa_col] | match_result_qa.loc[match_result.index]
        match_result["has_qa"] = match_result[self.qa_map.values()].any(axis=1)
        return match_result
        
    def check_for_qa_cols(self):
        """
            This function ensures that there is qa flag for each column that is processed. 
            In case there is missing QA column, either specify a qa_map during on_person_cols,
            define a mapping with define_qa_pairs or add a qa column to the data. Deduplication only runs 
            if self.every_col_has_qa = True. Set is manually if qa should be ignored.
        """
        qa_cols = [col for col in self.data.columns if re.search(r"_qa$",col)]
        cols_without_qa_col = []
        match_cols = {*self.person_cols, *self.date_cols, *self.other_cols, *self.other_strict_cols}
        for col in match_cols:
            if col not in self.qa_map:
                col_qa = col
                while f"{col_qa}_qa" not in qa_cols and col_qa != "":
                    if not re.search(r"_[\da-zA-Z]+$",col_qa):
                        col_qa = ""
                    col_qa = re.sub(r"_[\da-zA-Z]+$","",col_qa)
                col_qa = f"{col_qa}_qa"
                if col_qa in qa_cols:
                    self.qa_map[col] = col_qa
                else: 
                    cols_without_qa_col.append(col)
                    
        if len(cols_without_qa_col) > 0:
            print(f"WARNING: No QA Column found for:\n{"\n".join(cols_without_qa_col)}")
            self.every_col_has_qa = False
        else:
            self.every_col_has_qa = True

    def define_qa_pairs(self,qa_map: dict):
        """
            The qa_map dict defines a mapping from columns that should be deduplicated and their respective qa columns.
            The mapping does not need to be injective.
        """
        self.qa_map = {**self.qa_map, **qa_map}
        self.check_for_qa_cols()
        return self

    def on_person_cols(self,person_cols,qa_map=None):
        """
            Defines columns with person information that should be deduplicated.
            Person columns use the Default_Person_Col_Matcher per default. Can be 
            changed with set_col_matcher. Object_ids UUIDS are only generated if person
            columns are defined.
        """
        self.person_cols = person_cols
        if not qa_map is None:
            self.define_qa_pairs(qa_map)
        self.check_for_qa_cols()
        return self

    def on_date_cols(self,date_cols,qa_map=None):
        """
            Defines columns with date information that should be deduplicated.
            date columns use the Default_Date_Col_Matcher per default. Can be 
            changed with set_col_matcher.
        """
        self.date_cols = date_cols
        if not qa_map is None:
            self.define_qa_pairs(qa_map)
        self.check_for_qa_cols()
        return self

    def on_other_cols(self,other_cols,qa_map=None):
        """
            Defines columns that should be deduplicated.
            date columns use the Default_Col_Matcher per default. Can be 
            changed with set_col_matcher. (Intended for Names or other text fields)
        """
        self.other_cols = other_cols
        if not qa_map is None:
            self.define_qa_pairs(qa_map)
        self.check_for_qa_cols()
        return self
        
    def on_other_strict_cols(self,other_strict_cols,qa_map=None):
        """
            Defines columns that should be deduplicated.
            date columns use the Default_Strict_Col_Matcher per default. Can be 
            changed with set_col_matcher. (Intended for IDS, Numbers etc)
        """
        self.other_strict_cols = other_strict_cols
        if not qa_map is None:
            self.define_qa_pairs(qa_map)
        self.check_for_qa_cols()
        return self
    
    def on_fuzzy_cols(self,fuzzy_cols,qa_map=None):
        """
            
        """
        self.fuzzy_cols = fuzzy_cols
        if not qa_map is None:
            self.define_qa_pairs(qa_map)
        self.check_for_qa_cols()
        return self
    

    def set_col_matcher(self,col,col_matcher):
        """
            Override default matching behavious.
        """
        self.matcher = self.matcher.with_col_matcher(col,col_matcher)
        return self
        
    def set_missing_col_matchers_to_default(self):
        """
            Define default matching behavious.
        """
        cols_with_matcher = self.matcher.col_matcher.keys()
        for col in self.person_cols:
            if col not in cols_with_matcher:
                self.set_col_matcher(col,Default_Person_Col_Matcher())
        for col in self.date_cols:
            if col not in cols_with_matcher:
                self.set_col_matcher(col,Default_Date_Col_Matcher())
        for col in self.other_strict_cols:
            if col not in cols_with_matcher:
                self.set_col_matcher(col,Default_Strict_Col_Matcher())     
        for col in self.other_cols:
            if col not in cols_with_matcher:
                self.set_col_matcher(col,Default_Col_Matcher())
        for col in self.fuzzy_cols:
            if col not in cols_with_matcher:
                self.set_col_matcher(col, Default_Fuzzy_Col_Matcher())     
        return self

    def run(self):
        """
            Runs the deduplication job.
        """
        assert self.every_col_has_qa, f"Not every col that is matched has a qa column defined"
     
        print("Set non explicity matching strategies to defaults.")
        self.set_missing_col_matchers_to_default()
        print("Run preprocessing")
        self.deduplication_preprocess()
        print("Run matching")
        match_result = self.matcher.match()
        print("Run postprocessing")
        self.deduplication_postprocess()

        # set old rows as deleted (artifact of old inplace deduplication script)
        print("Set deleted column")
        self.data['deleted'] = True
        match_result['deleted'] = False

        # set ambiguous column for old rows
        print("Set ambiguous column for old rows")
        self.data["is_ambiguous"] = match_result.loc[self.data[self.id_col]]["is_ambiguous"].values
        self.data["ambiguous_columns"] = match_result.loc[self.data[self.id_col]]["ambiguous_columns"].values
        
        # assing has_qa from preprocessing to match results
        print("Assing qa values to match results")
        match_result = self.assign_qa_flags_to_matching_result(match_result)

        # set object_ids if there are persons on documents
        if len(self.person_cols) > 0:
            print("Assing UUIDS for persons")
            match_result['object_id'] = [str(uuid.uuid4()) for i in match_result.index]
        else:
            print("No Person Columns present. Set empty UUIDS")
            match_result['object_id'] = np.full((match_result.shape[0],), np.nan)
        self.data["object_id"] = match_result.loc[self.data[self.id_col]]["object_id"].values

        if self.metadata_columns is not None:
            print("Copy Metadata from old rows")
            for col in self.metadata_columns:
                match_result[col] = match_result.index.to_series().apply(lambda idx: self.data[self.data[self.id_col] == idx][col].iloc[0])
        
        print("Attach results")
        match_result = match_result.reset_index()
        deduplicated_data = pd.concat([self.data, match_result],axis="rows")
        
        percent_amb = round(match_result.loc[match_result['is_ambiguous']].shape[0] / match_result.shape[0] * 100)
        print(f"{match_result.shape[0]} processed")
        print(f"{percent_amb} percent are ambiguous")

        self.deduplicated_data = deduplicated_data.fillna('').replace(r'\.0$', '', regex=True)
        return self.deduplicated_data
