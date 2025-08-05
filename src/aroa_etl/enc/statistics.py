import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, HTML
from collections.abc import Iterable
from ..utils import value_is_not_empty_q

def __group_has_entries_for_col_q(group, col):
    return group[col].apply(value_is_not_empty_q).sum() >= 1

def __is_in_group_with_entries_for_col_q(df, id_col, groups, col):
    return pd.Series([__group_has_entries_for_col_q(groups.get_group((idx,)), col) for idx in df[id_col].values]).sum()
    
def __get_doc_group_with_data(df, id_col):
    groups = df.groupby([id_col])
    return pd.Series([__is_in_group_with_entries_for_col_q(df, id_col, groups, col) for col in df.columns])

def __get_doc_with_data(df):
    return pd.Series([df[col].apply(value_is_not_empty_q).sum()#*100/df.shape[0]
                      for col in df.columns])

def df_has_data_stats(df,id_col):
    col_series = pd.Series([c for c in df.columns])
    doc_group_with_data = __get_doc_group_with_data(df,id_col)
    docs_with_data = __get_doc_with_data(df)
    no_values = np.full(col_series.shape,df.shape[0])-doc_group_with_data
    stats_df = pd.concat([col_series, no_values, docs_with_data, doc_group_with_data,], axis="columns")
    stats_df.columns = ["field","No-Values","Has Data","Group Has Data"]
    return stats_df.sort_values(by='Has Data', ascending=False)


def plot_has_data_stats(stats_df):
    num_entries = stats_df["No-Values"][0] + stats_df["Group Has Data"][0]
    bar_data = {
        "No-Values": np.array([stats_df.loc[i,"No-Values"] for i in stats_df.index]),
        "Part of Group with Values": np.array([stats_df.loc[i,"Group Has Data"] - stats_df.loc[i,"Has Data"] for i in stats_df.index]), 
        "Has Values": np.array([stats_df.loc[i,"Has Data"] for i in stats_df.index]),
    }
    bar_data_percent = {
        "No-Values":  np.array([round(bar_data["No-Values"][i]*100/num_entries,2) for i in range(stats_df.shape[0])]),
        "Part of Group with Values":  np.array([round(bar_data["Part of Group with Values"][i]*100/num_entries,2) for i in range(stats_df.shape[0])]), 
        "Has Values": np.array([round(bar_data["Has Values"][i]*100/num_entries,2) for i in range(stats_df.shape[0])]), 
    }
    color_map = {
            "No-Values": "gray",
            "Part of Group with Values": "orange", 
            "Has Values": "green", 
    }
    hover_text = {
            "No-Values" : ["<br>".join([f"<b>No-Values:  {bar_data["No-Values"][idx]}  ({bar_data_percent["No-Values"][idx]}%)</b>",
                              f"Group has Values:  {bar_data["Part of Group with Values"][idx]}  ({bar_data_percent["Part of Group with Values"][idx]}%)",
                              f"Has Values:  {bar_data["Has Values"][idx]}  ({bar_data_percent["Has Values"][idx]}%)"
                             ])
                   for idx, c in enumerate(stats_df.field)],
            "Part of Group with Values" :  ["<br>".join([f"No-Values:  {bar_data["No-Values"][idx]}  ({bar_data_percent["No-Values"][idx]}%)",
                              f"<b>Group has Values:  {bar_data["Part of Group with Values"][idx]}  ({bar_data_percent["Part of Group with Values"][idx]}%)</b>",
                              f"Has Values:  {bar_data["Has Values"][idx]}  ({bar_data_percent["Has Values"][idx]}%)"
                             ])
                   for idx, c in enumerate(stats_df.field)],
            "Has Values" : ["<br>".join([f"No-Values:  {bar_data["No-Values"][idx]}  ({bar_data_percent["No-Values"][idx]}%)",
                              f"Group has Values:  {bar_data["Part of Group with Values"][idx]}  ({bar_data_percent["Part of Group with Values"][idx]}%)",
                              f"<b>Has Values:  {bar_data["Has Values"][idx]}  ({bar_data_percent["Has Values"][idx]}%)</b>"
                             ])
                   for idx, c in enumerate(stats_df.field)]
    }
    fig = go.Figure()
    for i, stat in enumerate(["Has Values", "Part of Group with Values", "No-Values"]):
        fig.add_trace(go.Bar(
                x=[stats_df.loc[i,"field"] for i in stats_df.index],
                y=bar_data_percent[stat],
                name=stat,
                offsetgroup=i,
                marker_color=[color_map[stat] for i in range(0,len(bar_data[stat]))],
                hovertext=hover_text[stat]
        ))

    fig.update_layout(
            barmode='stack',
            title='Annotations with non-empty data',
            xaxis_title='Transcribed Fields',
            yaxis_title='Percent'
    )
    display(HTML(fig.to_html(full_html=False)))