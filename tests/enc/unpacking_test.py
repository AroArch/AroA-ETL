import pytest
import sys
sys.path.insert(0, 'src')

import pandas as pd
import re
from aroa_etl.enc.unpacking import unpack
      
def test_unpack_data():
    raw_data = pd.read_csv("testing_data/raw_enc_data.csv",index_col=0)
    unpacked_data = unpack(raw_data,"json_data", additional_splits_on=lambda col: re.search(r"(category)",col), split_re=r"[\|;,\s]")
    assert unpacked_data.columns.tolist() == ['workflow_id', 'document_id', 'prisoner_category_0',
                                              'prisoner_category_1', 'prisoner_category_2', 'prisoner_category_3',
                                              'prisoner_category_4', 'prisoner_category_5', 'prisoner_number',
                                              'imprisonment_year', 'imprisonment_month', 'imprisonment_day',
                                              'imprisonment_camp', 'place_of_birth_0', 'place_of_birth_1',
                                              'birthdate_year', 'birthdate_month', 'birthdate_day', 'first_name_0',
                                              'first_name_1', 'last_name'], "Columns mismatch after unpacking"
    assert unpacked_data.first_name_0.tolist() == ['Dr.Alice', 'Dr. Alice', 'Alice', 'Bob1 Bob2', 'Bob1', 'Bob1', 'Bob1', 'Bob1'], "Unpexpected outcome for unpacking first_names"
    #unpacked_data.to_csv("testing_data/unpacked_enc_data.csv")
    
