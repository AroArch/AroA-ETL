import pytest
import sys
sys.path.insert(0, 'src')

import pandas as pd
import re
from aroa_etl.enc.processing import process_unpacked_data
      
def test_prossess_unpacked_data():
    unpacked_data = pd.read_csv("testing_data/unpacked_enc_data.csv",index_col=0,dtype=str)
    processed_data = process_unpacked_data(unpacked_data,
                                       skip_columns=[
                                           'updated_at',
                                           'user_id',
                                           'workflow_id', 
                                           'created_at', 
                                           'document_id', 
                                           'id'                                           
                                       ],)
    assert processed_data.imprisonment_camp_qa.sum() == 5, "QA columns are not assigned properly"
    assert "imprisonment_camp_data_source" in processed_data.columns, "Data Source column is missing"
    assert processed_data.last_name_cleaned_0[:5].tolist() == ["Muller", "Müller", "Mueller", "Schmidt", "Schmïdt"] , "Last Name Processing did not work"
    assert processed_data.first_name_cleaned_0.tolist() == ['Alice', 'Alice', 'Alice', 'Bob1', 'Bob1', 'Bob1', 'Bob1', 'Bob1'], "First Name Processing did not work"
    #processed_data.to_csv("testing_data/normalised_enc_data.csv")
