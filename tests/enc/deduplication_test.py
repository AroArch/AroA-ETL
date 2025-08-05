import pytest
import sys
sys.path.insert(0, 'src')

import pandas as pd
import re
from aroa_etl.enc.deduplication import ENC_Deduplicater
      
def test_prossess_unpacked_data():
    processed_data = pd.read_csv("testing_data/normalised_enc_data.csv",index_col=0,dtype=str)
    person_cols = [
         'first_name_cleaned_0',
         'first_name_cleaned_1',
         'last_name_cleaned_0',
    ]
    date_cols = [ 
         'birthdate_day_cleaned',
         'birthdate_month_cleaned',
         'birthdate_year_cleaned',
         'imprisonment_day_cleaned',
         'imprisonment_month_cleaned',
         'imprisonment_year_cleaned',
    ]
    other_cols = [
        'imprisonment_camp_cleaned',
        'place_of_birth_0_cleaned',
        'place_of_birth_1_cleaned',
    ]
    other_strict_cols = [ 
        'prisoner_category_0_cleaned',
        'prisoner_category_1_cleaned',
        'prisoner_category_2_cleaned',
        'prisoner_category_3_cleaned',
        'prisoner_category_4_cleaned',
        'prisoner_category_5_cleaned'
    ]
    deduplicater = ENC_Deduplicater(processed_data,"document_id",metadata_columns=["object_id", "workflow_id"])#"id", "created_at", "updated_at", "user_id",
    deduplicater.on_person_cols(person_cols)
    deduplicater.on_date_cols(date_cols)
    deduplicater.on_other_cols(other_cols)  
    deduplicater.on_other_strict_cols(other_strict_cols) 

    deduplication_result = deduplicater.run()
    
    assert "object_id" in deduplication_result.columns, "object id not assigned"

    assert deduplicater.matcher.match()["last_name_cleaned_0"].to_list() == ["Müller", "Schmidt"], "Last Name Deduplication did not work"
    assert deduplicater.matcher.match()["first_name_cleaned_0"].to_list() == ["Alice", "Bob1"], "First Name Deduplication did not work"
    assert deduplicater.matcher.match()["first_name_cleaned_1"].to_list()[1] == "Bob2", "First Name Deduplication did not work"
    assert deduplicater.matcher.match()["place_of_birth_0_cleaned"].to_list() == ["Frankfurt", "Stadt"], "First Name Deduplication did not work"
    assert set(deduplication_result.ambiguous_columns.values[0].split(', ')) == {'imprisonment_year_cleaned', 'prisoner_category_5_cleaned', 'imprisonment_camp_cleaned', 'place_of_birth_1_cleaned'}, "Got different ambiguous columns"
    
    #deduplication_result.to_csv("testing_data/marked_enc_data.csv")

