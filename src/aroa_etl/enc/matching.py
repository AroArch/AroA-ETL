import pandas as pd
import re
import numpy as np
import unicodedata
from iteration_utilities import first
from itertools import zip_longest
from collections.abc import Iterable
from collections import Counter
from jellyfish import jaro_similarity
import plotly.express as px
from IPython.display import display, HTML
import plotly.graph_objects as go
from ..utils import value_is_not_empty_q, replace_special_character, replace_umlaut_character, has_value_q
from rapidfuzz import fuzz, utils

class Col_Matcher():
    def __init__(self,):
        self.match_pipeline = []

    def __to_ascii(name):
        """
            Convert String to ascii characters only (also converts ü to u).
        """
        name = replace_special_character(name)
        return unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('UTF-8', 'ignore')

    def on_ascii(self):
        """
            Ignores non-ascii characters during matching.
        """
        self.match_pipeline.append(lambda enc_doc: enc_doc.apply(Col_Matcher.__to_ascii))
        return self
    
    def __to_ascii_with_umlaut(name):
        """
            Convert String to ascii characters only (ignoring üöäß).
        """
        return "".join(Col_Matcher.__to_ascii(c) if not re.match("[äöüß]",c) else c for c in name )
        
    def on_ascii_with_umlaut(self):
        """
            Ignores non-ascii characters (except german umlaute) during matching.
        """
        self.match_pipeline.append(lambda enc_doc: enc_doc.apply(Col_Matcher.__to_ascii_with_umlaut))
        return self
    
    def __substitute_umlaute(name):
        """
            Convert german umlate and other symbols. (ö to oe and so on).
        """
        return replace_umlaut_character(name)
    
    def __to_ascii_with_umlaut_normalized(name):
        """
            First substitutes umlaute (ö to oe and so on) and converts remaining characters to ascii characters.
        """
        name = Col_Matcher.__substitute_umlaute(name)
        name = Col_Matcher.__to_ascii_with_umlaut(name)
        return name

    def on_ascii_with_umlaut_normalized(self):
        """
            Ignores non-ascii characters during matching. German umlaute are converted (ü -> ue, ...).
        """
        self.match_pipeline.append(lambda enc_doc: enc_doc.apply(Col_Matcher.__to_ascii_with_umlaut_normalized))
        return self

    def __complete_known_abbreviations(name):
        """
            Completion of known abbreviations. Designed for street/location fields.
        """
        abbreviation = {r"(?P<str>[sS]tr)a?\.": r"\g<str>aße",
                        r"(?P<str>[sS]tr)a?$": r"\g<str>aße",
                        r"\sb\.": r" bei", 
                        r"\s[kK]rs?\.?\s?": " Kreis ",
                        r"(?P<sep1>[^\w])[Bb]ln\.?(?P<sep2>[\s\-=])":r"\g<sep1>Berlin\g<sep2>",
                        r"^[Bb]ln\.?(?P<sep>[\s\-=])":r"Berlin\g<sep>",
                        r"(?P<sep1>[^\w])[lL][kK]r?[\.\s]": " Landkreis ",
                        r"(?P<number>\d+)(?P<letter>[a-zA-Z])": r"\g<number> \g<letter>"
                       }
        for abb in abbreviation:
            name = re.sub(abb,abbreviation[abb],name)
        return name

    def with_known_abbreviations_completed(self):
        """
            Completes known abbreviations before matching.
        """
        self.match_pipeline.append(lambda enc_doc: enc_doc.apply(Col_Matcher.__complete_known_abbreviations))
        return self
    
    def __customSyllableMatcherCol(names,word_col):
        """
            Performs a windowed/syllable matching of names.
    
            Example:
                  Frankfurt
                  Frankfurter
                  Frandfurt
    
            Is matched to Frankfurt. In case there is a matching, the changes are updated inplace in the names list.
        """
        # Test if there are at least 3 names
        if len(word_col)<3:
            return names
        # test if the column is about the same word
        for w1, w2 in zip(word_col,word_col[1:]+word_col[:1]):
                if w1!=None and w2!=None and jaro_similarity(w1,w2) < 0.8:
                    return names #nothing changes
    
        # voting for each word
        word_scores = np.zeros(len(word_col))
        for word_idx, word in enumerate(word_col):
            other_words = word_col[:word_idx]+word_col[word_idx+1:]
            window_len = 3
            if word == None or len(word)<window_len:
                continue
            score_name = np.zeros(len(word)+1-window_len)
            for window_start in range(len(word)+1-window_len):
                window = word[window_start:window_start+window_len]
                for oword in other_words:
                    if oword!= None and window in oword and abs(oword.index(window) - window_start)<3:
                        score_name[window_start] += 1
            word_scores[word_idx] += 0 if score_name.min() == 0 else score_name.mean()
        best = word_scores.argmax()
    
        # 1 means one other (2 with self vote)
        if word_scores[best] != 0:
            for word_idx, word in enumerate(word_col):
                if word != None:
                    names.iloc[word_idx] = names.iloc[word_idx].replace(word,word_col[best])
        return names

    def __customSyllableMatcher(enc_doc):
        """ 
            Syllable matcher for a pd.series of names. Names can be sentances for which the syllable is performed for each column.
            
            
            Example:
                  [
                  "Word1 ... Frankfurt word10",
                  "Word1 ... Frankfurter word10",
                  "Word1 ... Frandfurt word10",
                  ]
    
            Performs the syllable matching for [Word1,Word1,Word1] ,... [Frankfurt, Frankfurter, Frandfurt] and [word10,word10,word10].
            The matching result for column 9 would be Frankfurt. 
    
            Returns: Names with matchings updated inplace.
        """
        enc_line_idx = enc_doc.index
        names = list(enc_doc)
        words_at_equal_pos = zip_longest(*[re.findall(r"[\w\.]+",name) for name in enc_doc])
        for word_col in words_at_equal_pos:
            # updates inplace
            names = Col_Matcher.__customSyllableMatcherCol(enc_doc,word_col)
        return pd.Series(enc_doc,index=enc_line_idx)

    def with_syllable_matching(self):
        """
            Enables for a syllable inspired matching strategie. 
        """
        self.match_pipeline.append(Col_Matcher.__customSyllableMatcher)
        return self
    
    def __customFuzzyMatcher(enc_doc):
        enc_doc = enc_doc.astype(str)
        enc_doc = enc_doc.loc[enc_doc.apply(has_value_q)]
        if len(enc_doc) == 0:
            return "-"
        median = np.array([
            np.array([
                fuzz.ratio(value,other_value,processor=utils.default_process) 
                for other_value in enc_doc
            ]).mean()
            for value in enc_doc
        ]).argmax()
        return enc_doc.iat[median]

    def with_fuzzy_matching(self):
        self.match_pipeline.append(Col_Matcher.__customFuzzyMatcher)
        return self

    def __substritude_all(name, substitution_map):
        """
            Small helper method that applies a map of text substitutions.
        """
        for subs, replacement in substitution_map.items():
            name = name.replace(subs,replacement)
        return name

    def with_custom_substitution(self,pattern,subs):
        """
            Adds a custom substitution during matching.
        """
        self.match_pipeline.append(lambda enc_doc: enc_doc.apply(lambda entry: re.sub(pattern,subs,entry)))
        return self

    def with_custom_replace(self,pattern,repl):
        """
            Replaces an entire field by repl if pattern is found.
        """
        self.match_pipeline.append(lambda enc_doc: enc_doc.apply(lambda entry: repl if re.search(pattern,entry) else entry))
        return self
        
    def __abbreviation_completions(enc_doc):
        """
            Tests if there is one entry that completed an abbreveation and applies that to all (inplace).
            Input are all enc entries for a single field. 
        """
        abbreviations = [(pos,word) for entry in enc_doc 
                                         for pos, word in enumerate(re.findall(r"[\w\.]+",entry))
                                         if re.match(r"\w{3,}\.",word) # is abbreviation
                        ]
        
        complete_abbreviations = dict()
        for pos, abbreviation in abbreviations:
            for entry in enc_doc:
                words = re.findall(r"[\w\.]+",entry)
                if len(words)<=pos:
                    continue
                word_in_other_entry = words[pos]
                if "." not in word_in_other_entry and len(word_in_other_entry) > len(abbreviation)+1 and word_in_other_entry[0] == abbreviation[0]:
                    complete_abbreviations[abbreviation] = word_in_other_entry

        enc_doc = enc_doc.apply(lambda name: Col_Matcher.__substritude_all(name, complete_abbreviations))
        return enc_doc

    def with_automatic_abbreviation_completion(self):
        """
            Enables an automatic completion of abbreviations.
        """
        self.match_pipeline.append(Col_Matcher.__abbreviation_completions)
        return self

    def __automatic_umlaut_substitution(enc_doc):
        """
            Tests if there is one entry that interpreteded a symbol as an umlaut. 
        """
        # equal except for umlate (ignores casing)
        umlaut_words = [(pos,word) for entry in enc_doc 
                                         for pos, word in enumerate(re.findall(r"[\w\.]+",entry)) 
                                         if re.search(r"[üöäß]",word)]
        umlaut_substitutions = dict()
        for entry in enc_doc:
            for pos, umlaut_word in umlaut_words:
                entry_words = re.findall(r"[\w\.]+",entry)
                if len(entry_words)<=pos:
                    continue
                candidate = entry_words[pos]
                if len(candidate) >= len(umlaut_word) and \
                   (Col_Matcher.__to_ascii_with_umlaut(umlaut_word.lower()) == Col_Matcher.__to_ascii_with_umlaut(candidate.lower())
                   or Col_Matcher.__to_ascii(umlaut_word.lower()) == Col_Matcher.__to_ascii(candidate.lower())
                   or Col_Matcher.__substitute_umlaute(umlaut_word.lower()) == Col_Matcher.__substitute_umlaute(candidate.lower())):
                         umlaut_substitutions[candidate] = umlaut_word
    
        enc_doc = enc_doc.apply(lambda name: Col_Matcher.__substritude_all(name, umlaut_substitutions))
        return enc_doc

    def with_automatic_umlaut_substitution(self):
        """
            Enables an automatic substitution of characters to umlaut if and only if one entry supports it.
        """
        self.match_pipeline.append(Col_Matcher.__automatic_umlaut_substitution)
        return self

    def __capitalization_substitution(enc_doc):
        """
            Automatically capitalizes words if one entry says so.
        """
        upper_case_words = [(pos,word) for entry in enc_doc 
                                         for pos, word in enumerate(re.findall(r"[\w\.]+",entry)) 
                                         if re.match(r"[A-Z]\w*",word)]
        capitalization_substitution = dict()
        for entry in enc_doc:
            for pos, upper_case_word in upper_case_words:
                entry_words = re.findall(r"[\w\.]+",entry)
                if len(entry_words)<=pos:
                    continue
                candidate = entry_words[pos]
                if candidate != upper_case_word and candidate.lower() == upper_case_word.lower():
                    capitalization_substitution[candidate] = upper_case_word
    
        enc_doc = enc_doc.apply(lambda name: Col_Matcher.__substritude_all(name, capitalization_substitution))
        return enc_doc
        
    def with_automatic_capitalization_substitution(self):
        """
            Enables automatic capitalization of words. (... only if one entry supports it)
        """
        self.match_pipeline.append(Col_Matcher.__capitalization_substitution)
        return self

    def __match_doc(enc_doc):
        """
        Mappes a list of strings onto one of them iff each word is supported by at least one other string.
        Input are enc lines as pd.Series.

        Example:
        __match(pd.Series(["one two","one tw", "on two"]))
        """
        voting = []
        # match on each word individually
        match_strings = [re.findall(r"([a-zA-ZäöüßÄÜÖ]+\.?|\d+)",entry) for entry in enc_doc]
        match_strings = [entry_words for entry_words in match_strings if len(entry_words)>0] # remove empty entries
        
        # no match 
        len_count = Counter(len(m) for m in match_strings)
        if not [i for i in len_count.values() if i > 1]:
            return np.nan
        
        for pos_a, entry_a_words in enumerate(match_strings):
            score_a = np.zeros(len(entry_a_words))
            for pos, entry_a_word in enumerate(entry_a_words):
                for entry_b_word in [entry_b_word for entry_b_words in match_strings for entry_b_word in entry_b_words]:
                    if entry_b_word in entry_a_word:
                        score_a[pos] += 1
            voting.append((pos_a,score_a.min()))
    
        match_pos, match_count = sorted([(pos,score) for pos,score in voting if len_count[len(match_strings[pos])] > 1],key=lambda tpl: tpl[1])[-1]
        match = enc_doc.values[match_pos] if match_count>1 else np.nan 
        return match if match != "" else np.nan

    def break_if(self, condition, except_value):
        """
            Break a matching if the condition applies. Returns np.nan for the currently matched document and column.
        """
        self.match_pipeline.append(lambda enc_doc: except_value if condition(enc_doc) else enc_doc)
        return self

    def exlude_empty(self,):
        def empty_excluder(enc_doc):
            if isinstance(enc_doc, pd.core.frame.DataFrame):
                assert enc_doc.shape[1] > 1, "only one attribute can be matched at a time"
                enc_doc = enc_doc[0]
            non_empty_doc = enc_doc[enc_doc.apply(lambda val: value_is_not_empty_q(val) and not re.match("[uU]nklar|[uU]nclear", val))]
            if len(non_empty_doc) < 2:
                return "-"
            return non_empty_doc
        self.match_pipeline.append(empty_excluder)

    def __call__(self,enc_doc):
        """
            Runs the matching for a document and column with all enabled functions. Functions are applied in order of their activation. 
        """
        pipeline = [*self.match_pipeline, Col_Matcher.__match_doc]
        for step in pipeline:
            if not isinstance(enc_doc, pd.core.series.Series) and not isinstance(enc_doc, pd.core.frame.DataFrame):
                return enc_doc
            enc_doc = step(enc_doc)
        return enc_doc #Col_Matcher.__match_doc(enc_doc)

class Default_Col_Matcher(Col_Matcher):
    """
        Default matcher for columns with all features.
        Intended for text based columns like names.
    """
    def __init__(self):
        super().__init__()
        self.exlude_empty()
        #self.break_if(lambda enc_doc: 1<len([name for name in enc_doc if re.match(r"[\-\s]+$",name)]),"-")
        #self.break_if(lambda enc_doc: re.match(r"\-+",first(enc_doc.value_counts().items())[0]),"-")  
        self.with_custom_substitution(r"\s+",r" ").with_custom_substitution(r"\s(?P<sym>[^a-zA-Z])\s",r"\g<sym>")
        self.with_automatic_umlaut_substitution().with_automatic_abbreviation_completion().on_ascii_with_umlaut().with_automatic_capitalization_substitution().with_syllable_matching() 

class Default_Strict_Col_Matcher(Col_Matcher):
    """
        Default matcher for columns with verbatim matching.
        Intended for IDs or numbers.
    """
    def __init__(self):
        super().__init__()
        self.exlude_empty()
        #self.break_if(lambda enc_doc: 1<len([name for name in enc_doc if re.match(r"[\-\s]+$",name)]),"-")
        #self.break_if(lambda enc_doc: re.match(r"\-+",first(enc_doc.value_counts().items())[0]),"-")  

class Default_Person_Col_Matcher(Col_Matcher):
    """
        Default matcher for person fields.
        Currently all features are enabled.
    """

    def __init__(self):
        super().__init__()
        self.exlude_empty()
        #self.break_if(lambda enc_doc: 1<len([name for name in enc_doc if re.match(r"[\-\s]+$",name)]),"-")
        #self.break_if(lambda enc_doc: re.match(r"\-+",first(enc_doc.value_counts().items())[0]),"-")  
        self.with_custom_substitution(r"\s+",r" ").with_custom_substitution(r"\s(?P<sym>[^a-zA-Z])\s",r"\g<sym>")
        self.with_automatic_umlaut_substitution().with_automatic_abbreviation_completion().on_ascii_with_umlaut().with_automatic_capitalization_substitution().with_syllable_matching()

class Default_Date_Col_Matcher(Col_Matcher):
    """
        Default matcher for date fields.
        Currently no feature is enabled. (Verbatim matching)
    """
    
    def __init__(self):
        super().__init__()
        self.break_if(lambda enc_doc: 1<len([name for name in enc_doc if re.match(r"[\-\s]+$",name)]),"-")
        self.break_if(lambda enc_doc: re.match(r"\-+",first(enc_doc.value_counts().items())[0]),"-")  

class Default_Fuzzy_Col_Matcher(Col_Matcher):
    def __init__(self):
        super().__init__()
        self.with_custom_substitution(r"\s+",r" ").with_custom_substitution(r"\s(?P<sym>[^a-zA-Z])\s",r"\g<sym>")
        self.with_automatic_umlaut_substitution().with_automatic_abbreviation_completion().on_ascii_with_umlaut().with_automatic_capitalization_substitution()
        self.with_fuzzy_matching()

class Enc_Matcher():
    """
    Example usage:
    
    >>> data.loc[data.aroa_doc_id.notna()]
    >>> matcher = Enc_Matcher(data,'aroa_doc_id')\
                           .combine_columns(['last_adress_city_0','last_adress_city_1'],'full_city_field',
                                                              sep=', ',join_filter=lambda v: pd.notna(v) and re.search(r'[a-zA-Z]',v))\
                           .with_col_matcher('full_city_field',
                                              Col_Matcher().break_if(lambda enc_doc: 1<len([name for name in enc_doc if re.match(r'[\-\s]+$',name)]),'-')\
                                                           .break_if(lambda enc_doc: re.match(r'\-+',first(enc_doc.value_counts().items())[0]),'-')\
                                                           .with_custom_replace('deutschland','Hat nie in Deutschland gelebt')\
                                                           .with_custom_substitution(r'\s(?P<number>[XVI]+)\.', r' \g<number>')\ 
                                                           .with_custom_substitution(r'\s+',r' ')\                              
                                                           .with_custom_substitution(r'\s(?P<sym>[^a-zA-Z])\s', r'\g<sym>')\      
                                                           .with_custom_substitution('=','-')\
                                                           .with_automatic_umlaut_substitution()\
                                                           .with_automatic_abbreviation_completion()\
                                                           .on_ascii_with_umlaut()\
                                                           .with_automatic_capitalization_substitution()\
                                                           .with_syllable_matching()  )\
                           .combine_columns(['place_of_birth_0','place_of_birth_1'],'place_of_birth',
                                                              sep=', ',join_filter=lambda v: pd.notna(v) and re.search(r'[a-zA-Z]',v))\
                           .with_col_matcher('place_of_birth',
                                              Col_Matcher().break_if(lambda enc_doc: 1<len([name for name in enc_doc if re.match(r'[\-\s]+$',name)]),'-')\
                                                           .break_if(lambda enc_doc: re.match(r'\-+',first(enc_doc.value_counts().items())[0]),'-')\
                                                           .with_custom_replace('deutschland','Hat nie in Deutschland gelebt')\
                                                           .with_custom_substitution(r'\s(?P<number>[XVI]+)\.', r'\g<number>')\ 
                                                           .with_custom_substitution(r"\s+",r" ")\                              
                                                           .with_custom_substitution(r'\s(?P<sym>[^a-zA-Z])\s', r'\g<sym>')\      
                                                           .with_custom_substitution('=','-')\
                                                           .with_automatic_umlaut_substitution()\
                                                           .with_automatic_abbreviation_completion()\
                                                           .on_ascii_with_umlaut()\
                                                           .with_automatic_capitalization_substitution()\
                                                           .with_syllable_matching()  )
    """
    def __init__(self,enc_data,id_col):
        self.enc_data = enc_data
        self.id_col=id_col
        self.col_matcher = dict()
        self.log = ""
        self.match_result = None
        self.data_groups = None 
        self.stats_df = None

    def __get_data_groups(self):
        if self.data_groups == None:
            self.data_groups = self.enc_data.groupby([self.id_col])
        return self.data_groups

    def __id_has_entries_for_col_q(self, idx, col, numeric=False):
        """
            Small helper method that checks if there is a non empty entry for a document and column.
        """
        num_entries = self.__get_data_groups().get_group((idx,))[col].apply(value_is_not_empty_q).sum()
        if numeric:
            return num_entries
        return num_entries >= 1
    
    def combine_columns(self, columns, new_col_name, sep=", ",join_filter=pd.notna):
        """
            Combines columns before matching.
        """
        self.enc_data[new_col_name] = self.enc_data[columns].apply(lambda row: sep.join([field for field in row if join_filter(field)]),axis=1)
        return self

    def with_col_matcher(self,col,col_matcher=None):
        """
            Defines how a column is matched. Uses Default_Col_Matcher 
            (intended for text based columns) per default.
        """
        if col_matcher:
            self.col_matcher[col] = col_matcher
        else:
            self.col_matcher[col] = Default_Col_Matcher()
        return self

    def successful_matches(self, cols=None, no_values_is_a_match=False):
        """
            Returns a Boolean pd.Series that encodes if all columns in cols are successfully matched for a document.
        """
        if cols is None:
            cols = list(self.col_matcher.keys())
        match_result = self.match()
        got_matched = np.full((self.match_result.shape[0],), True)
        for c in cols:
            col_got_matched = match_result[c].apply(value_is_not_empty_q) & (match_result[c] != "?")
            if no_values_is_a_match:
                col_got_matched = col_got_matched | match_result.index.to_series().apply(lambda idx: not self.__id_has_entries_for_col_q(idx,c))
            got_matched = got_matched & col_got_matched
        return got_matched 
    
    def is_ambiguous_col(self,cols=None,no_values_is_a_match=False):
        """
            Returns a Boolean pd.Series that encodes if there was a conflict 
            during the matching of a document for columns in cols.
        """
        match_result = self.match()
        not_matched = ~(self.successful_matches(cols,no_values_is_a_match=no_values_is_a_match))
        not_matched.index = not_matched.index.astype(str)
        return not_matched

    def raw_data_is_ambiguous_col(self,cols=None,no_values_is_a_match=False):
        not_matched = self.is_ambiguous_col(cols,no_values_is_a_match)
        return not_matched.loc[self.enc_data[self.id_col],:]

    def set_ambiguous_columns(self,match_result,no_values_is_a_match):
        """
            Adds a column to the matching results that contains a comma 
            separated list of columns that could not be matched. 
        """
        ambiguous_col = self.is_ambiguous_col(no_values_is_a_match=no_values_is_a_match)
        ambiguous_col.index = ambiguous_col.index.astype(str)
        match_result["is_ambiguous"] = ambiguous_col

        ambiguous_columns = []
        matched_columns = pd.Series(list(self.col_matcher.keys()))
        for col in matched_columns:
            print(f"   | Compute ambiguous col for {col}")
            ambiguous_columns.append(self.is_ambiguous_col(cols=[col],no_values_is_a_match=no_values_is_a_match))
        ambiguous_columns_col = pd.DataFrame(ambiguous_columns).apply(lambda amb_cols: ", ".join(matched_columns[amb_cols.values]),axis=0)

        match_result["ambiguous_columns"] = ambiguous_columns_col
        return match_result

    def set_unmatched_columns(self,match_result):
        for idx, ambiguous_cols in enumerate(match_result["ambiguous_columns"]):
            if ambiguous_cols != '':
                match_result.iloc[idx,match_result.columns.isin(ambiguous_cols.split(", "))] = "?"
        return match_result


    def match(self,no_values_is_a_match=True):
        """
            Executes the matching job. Results are cashed. 
            Returns a DataFrame with one row per document. 
            Contains np.nan if the match was conflicting or empty.
        """
        if not self.match_result is None:
            return self.match_result
        enc_docs = self.__get_data_groups()
        match_result = []
        for col in self.col_matcher.keys():
            print(f"   | Run matching for {col}")
            matching_result = enc_docs[col].apply(self.col_matcher[col])
            match_result.append(matching_result)

        match_result = pd.concat(match_result, axis=1)
        self.match_result = match_result
        
        print(f"Set ambiguous col for matching results")
        match_result = self.set_ambiguous_columns(match_result,no_values_is_a_match)
        
        print(f"Set unmatched entries to ?")
        match_result = self.set_unmatched_columns(match_result)

        self.match_result = match_result.fillna("")
        return self.match_result

    ### Methods for debugging
    
    def show_unmatched(self,cols=None):
        """
            Returns a dataframe with all documents that have 
            conflicting matches for one of cols.
        """
        if cols is None:
            cols = list(self.col_matcher.keys())
        self.is_ambiguous_col(cols)
        unmatched_rows = self.match()[self.is_ambiguous_col(cols).values].reset_index()[[self.id_col,*cols]]
        unmatched_ids = set(unmatched_rows[self.id_col].values)
        enc_data_filter = self.enc_data[self.id_col].apply(lambda idx: idx in unmatched_ids)
        return pd.concat([self.enc_data[enc_data_filter],unmatched_rows],axis="rows")

    def show_matched(self,cols=None):
        """
            Returns a dataframe with all documents are successfully matched.
        """
        if cols == None:
            cols = list(self.col_matcher.keys())
        self.is_ambiguous_col(cols)
        matched_rows = self.match()[(~self.is_ambiguous_col(cols)).values].reset_index()[[self.id_col,*cols]]
        matched_ids = set(matched_rows[self.id_col].values)
        enc_data_filter = self.enc_data[self.id_col].apply(lambda idx: idx in matched_ids)
        return pd.concat([self.enc_data[enc_data_filter],matched_rows],axis="rows")


    def stats(self,recompute=False):
        """
            Returns DataFrame with statistics about the matching results.
            
            Stat                      |    Col1   |    ...
            --------------------------+-----------+----------
            DocID with Entries        |           |
            DocID without Entries     |           |
            DocID Ambiguous Entries   |           |
            DocID Matched             |           |
            DocID with to few Entries |           |
        """
        if not self.stats_df is None and not recompute:
            return self.stats_df
        stats_list = []
        match_cols = list(self.col_matcher.keys())
        for c in match_cols:
            col_has_entries_num = self.match().index.to_series().apply(lambda idx: self.__id_has_entries_for_col_q(idx,c,numeric=True))

            no_values = (col_has_entries_num == 0)
            with_values = ~no_values
            num_col_without_entries = no_values.sum()
            
            is_matched = self.successful_matches(cols=[c], no_values_is_a_match=False) & with_values
            num_is_matched = is_matched.sum()
            
            num_is_ambiguous = ((~is_matched) & with_values).sum()
            
            num_col_with_entries = with_values.sum()
            num_col_not_enough_entries = (~is_matched & (col_has_entries_num == 1)).sum()
            num_is_ambiguous = num_is_ambiguous - num_col_not_enough_entries
            stats_list.append([num_col_with_entries, num_col_without_entries, num_is_ambiguous, num_is_matched, num_col_not_enough_entries])
        stats_df = pd.DataFrame(stats_list).T
        stats_df.columns = match_cols
        stats_df.index = ["DocID with Entries", "DocID without Entries", "DocID Ambiguous Entries", "DocID Matched", "DocID with to few Entries"]
        self.stats_df = stats_df
        return stats_df

    def stats_chart(self):
        """
            Compiles statistings about the matching results into a nice chart.
        """
        stats_df = self.stats().sort_values(by="DocID Matched", ascending=False,axis="columns")
        matches = self.match()
        bar_data = {
            "No-Values": np.array([stats_df.loc["DocID without Entries",c] for c in stats_df.columns]),
            "Matched": np.array([stats_df.loc["DocID Matched",c] #- stats_df.loc["DocID without Entries",c] 
                        for c in stats_df.columns]),
            "To-few-Values": np.array([stats_df.loc["DocID with to few Entries",c] for c in stats_df.columns]),
            "Ambiguous": np.array([stats_df.loc["DocID Ambiguous Entries",c] for c in stats_df.columns])
        }
        bar_data_percent = {
            "No-Values": np.array([round(stats_df.loc["DocID without Entries",c]*100/matches.shape[0],2) for c in stats_df.columns]),
            "Matched": np.array([round(stats_df.loc["DocID Matched",c] *100/matches.shape[0],2) #- stats_df.loc["DocID without Entries",c])
                        for c in stats_df.columns]),
            "To-few-Values": np.array([round(stats_df.loc["DocID with to few Entries",c]*100/matches.shape[0],2) for c in stats_df.columns]),
            "Ambiguous": np.array([round(stats_df.loc["DocID Ambiguous Entries",c]*100/matches.shape[0],2) for c in stats_df.columns])
        }
        color_map = {
            "No-Values": "lightgreen",
            "Matched": "darkgreen", 
            "Ambiguous": "red", 
            "To-few-Values": "orange",
        }
        hover_text = {
            "No-Values" : ["<br>".join([f"<b>No-Values:  {bar_data["No-Values"][idx]}  ({bar_data_percent["No-Values"][idx]}%)</b>",
                              f"Matched:  {bar_data["Matched"][idx]}  ({bar_data_percent["Matched"][idx]}%)",
                              f"Ambiguous(Few Values):  {bar_data["To-few-Values"][idx]}  ({bar_data_percent["To-few-Values"][idx]}%)",
                              f"Ambiguous:  {bar_data["Ambiguous"][idx]}  ({bar_data_percent["Ambiguous"][idx]}%)"
                             ])
                   for idx, c in enumerate(stats_df.columns)],
            "Matched" :  ["<br>".join([f"No-Values:  {bar_data["No-Values"][idx]}  ({bar_data_percent["No-Values"][idx]}%)",
                              f"<b>Matched:  {bar_data["Matched"][idx]}  ({bar_data_percent["Matched"][idx]}%)</b>",
                              f"Ambiguous(Few Values):  {bar_data["To-few-Values"][idx]}  ({bar_data_percent["To-few-Values"][idx]}%)",
                              f"Ambiguous:  {bar_data["Ambiguous"][idx]}  ({bar_data_percent["Ambiguous"][idx]}%)"
                             ])
                   for idx, c in enumerate(stats_df.columns)],
            "To-few-Values" : ["<br>".join([f"No-Values:  {bar_data["No-Values"][idx]}  ({bar_data_percent["No-Values"][idx]}%)",
                              f"Matched:  {bar_data["Matched"][idx]}  ({bar_data_percent["Matched"][idx]}%)",
                              f"<b>Ambiguous(Few Values):  {bar_data["To-few-Values"][idx]}  ({bar_data_percent["To-few-Values"][idx]}%)</b>",
                              f"Ambiguous:  {bar_data["Ambiguous"][idx]}  ({bar_data_percent["Ambiguous"][idx]}%)",
                            ])
                   for idx, c in enumerate(stats_df.columns)],
            "Ambiguous" : ["<br>".join([f"No-Values:  {bar_data["No-Values"][idx]}  ({bar_data_percent["No-Values"][idx]}%)",
                              f"Matched:  {bar_data["Matched"][idx]}  ({bar_data_percent["Matched"][idx]}%)",
                              f"Ambiguous(Few Values):  {bar_data["To-few-Values"][idx]}  ({bar_data_percent["To-few-Values"][idx]}%)",
                              f"<b>Ambiguous:  {bar_data["Ambiguous"][idx]}  ({bar_data_percent["Ambiguous"][idx]}%)</b>"
                             ])
                   for idx, c in enumerate(stats_df.columns)]
        }
        fig = go.Figure()
        for i, stat in enumerate(["No-Values","Matched","To-few-Values","Ambiguous"]):
            fig.add_trace(go.Bar(
                x=stats_df.columns,
                y=bar_data_percent[stat],
                name=stat,
                offsetgroup=i,
                marker_color=[color_map[stat] for i in range(0,len(bar_data[stat]))],
                hovertext=hover_text[stat]
            ))

        fig.update_layout(
            barmode='stack',
            title='Matching Results',
            xaxis_title='Matched Columns',
            yaxis_title='Percent'
        )
        display(HTML(fig.to_html(full_html=False)))

#    def deduplicate(self,include_raw=False,no_values_is_a_match=True):

#        match_result = self.match().copy()
#        
#        if not include_raw:
#            return match_result
#             
#        raw_data = self.enc_data.copy()
#        raw_data.columns = [f"{c} original" if c != self.id_col else c for c in raw_data.columns]
#        match_result = match_result.loc[raw_data[self.id_col],:]
#        raw_data = raw_data.drop(self.id_col,axis="columns")
#        raw_data_with_matches = pd.concat([raw_data,match_result.reset_index(drop=True)],axis="columns")
#        return raw_data_with_matches.sort_values(by=self.id_col, ascending=False,axis="rows")


