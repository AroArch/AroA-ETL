import json
from jsonschema import validate,Draft7Validator
import pandas as pd
from iteration_utilities import first

__task_with_additional_annotations_schema = {
    "type" : "object",
    "properties" : {
        "task" : {"type" : "string"},
         "name" : {"type" : "string"},
        'task_label': {"type" : "null"},
        "value" : {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"task": {"type": "string"}}
            }
        }
     },
}

__empty_task_schema = {"type": "object", 
              "properties": {"value": {"type": "array", "maxItems":1,"minItems":1, 
                                       "items": {"type": "object",
                                                 "additionalProperties": False,
                                                 "properties": {'select_label': {"type": "string"}}}}} }
__dropdown_task_schema = {"type": "object", 
              "properties": {"value": {"type": "array", "maxItems":1,"minItems":1, 
                                       "items": {"type": "object",
                                                 "properties": {
                                                     "value": {"type": "string"},
                                                     'select_label': {"type": "string"}}}}} }

__simple_task_schema = { "type": "object", "properties": {"value" : {"type": "string"}} }

def __process_task_with_additional_annotations(task_instance):
    parent_task = task_instance['task']
    additional_tasks = task_instance["value"]
    tasks = []
    while len(additional_tasks)!=0:
        next_task = additional_tasks.pop(0)
        if Draft7Validator(__simple_task_schema).is_valid(next_task):
            tasks.append(__process_simple_task(next_task))
        elif Draft7Validator(__empty_task_schema).is_valid(next_task):
            empty_task = additional_tasks.pop(0)
        elif Draft7Validator(__dropdown_task_schema).is_valid(next_task):
            value_task = additional_tasks.pop(0)
            tasks.append(__process_dropdown_task(next_task,value_task))
        else:
            assert False, f"neither task_types apply\n{task_instance}\n{next_task}"
    tasks_map = {task_name: task_value 
                 for task in tasks for task_name, task_value in task.items()}
    tasks_map = {": ".join([parent_task,task_name]): task_value 
                 for task_name, task_value in tasks_map.items()}
    return tasks_map

def __process_dropdown_task(task_type,task_value):
    assert len(task_type["value"])==1,f"task specifics to long\n{task_type}\n{task_value}"
    return {task_type["value"][0]["label"]:task_value["value"]}

def __process_simple_task(task_instance):
    assert len(task_instance["task_label"])>=1, f"simple task without label\n{task_instance}"
    return {task_instance["task_label"]:task_instance["value"]}

def __process_task(task_instance):
    if Draft7Validator(__task_with_additional_annotations_schema).is_valid(task_instance):
        return __process_task_with_additional_annotations(task_instance)
    elif Draft7Validator(__simple_task_schema).is_valid(task_instance):
        return __process_simple_task(task_instance)
    elif pd.isna(task_instance):
        return dict()
    assert False, f"not a valid instance\n{task_instance}"

def __process_task_list(task_list):
    return {task_name:task_value for task in task_list 
                                  for task_name, task_value in __process_task(task).items()
                                  if pd.notna(task)}

def __parse_subject(entry):
    if pd.isna(entry):
        return "",{}
    entry_json = json.loads(entry)
    assert len(entry_json.keys()) == 1, f"entry with multiple ids \n{entry_json}"
    return first(entry_json.keys()), first(entry_json.values())

def parse_annotations(annotation_series):
    annotations_df = pd.DataFrame.from_dict(annotation_series.apply(json.loads).values.tolist())
    annotations_df = pd.DataFrame.from_dict(annotations_df.apply(__process_task_list,axis="columns").values.tolist())
    return annotations_df
    
def parse_metadata(metadata_series):
    metadata_df = pd.DataFrame.from_dict(metadata_series.apply(json.loads).values.tolist())
    return metadata_df
    
def parse_subject_data(subject_series):
    subject_df = pd.DataFrame(subject_series.apply(__parse_subject).values.tolist(), columns=["Subject_ID","Subject_Data"])
    subject_data_df = pd.DataFrame.from_dict(subject_df["Subject_Data"].values.tolist())
    subject_df = subject_df.drop(["Subject_Data"],axis="columns")
    subject_df = pd.concat([subject_df,subject_data_df],axis="columns")
    subject_df.columns = [f"subject: {col}" for col in  subject_df.columns]
    return subject_df
    
def parse_zooniverse_data(df,drop_raw=True):
    annotations_df = parse_annotations(df.annotations)
    metadata_df = parse_metadata(df.metadata)
    subject_df = parse_subject_data(df.subject_data)
    if drop_raw:
        df = df.drop(["annotations","metadata","subject_data"],axis="columns")
    return df, annotations_df, metadata_df, subject_df