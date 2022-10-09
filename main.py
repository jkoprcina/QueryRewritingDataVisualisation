import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from processing_data import clean_data, split_df
st.set_page_config(layout="wide")

df = pd.read_csv("data/Query-Rewriting.csv")
df = clean_data(df)
st.write(df)
df_manual, df_testing_retriver_reranker, df_personal_testing_query_rewriting, df_personal_testing_everything, df_base_testing_query_rewriting, df_base_testing_everything = split_df(df)
df_dictionary = {
    'base_testing_query_rewriting': df_base_testing_query_rewriting,
    'base_testing_everything': df_base_testing_everything,
    'personal_testing_query_rewriting': df_personal_testing_query_rewriting,
    'personal_testing_everything': df_personal_testing_everything
}

col1, col2, col3 = st.columns(3)
with col1:
    T5_config = st.radio("What T5 config would you like to see?", ('base', 'personal'))
with col2:
    canonical = st.radio("Would you like to see canonical or non-canonical?",
                         ('testing_query_rewriting', 'testing_everything'))
with col3:
    score = st.radio("Would you like to see the bleu or rouge scores?", ('bleu', 'rouge', 'ndcg'))

st.subheader("You're looking at a graph showing the " + score + " score, ran on the T5 " + T5_config +
             " configuration ran " + canonical)

x = [i for i in range(2, 11)]
fig = go.Figure()
for index, row in df_dictionary.get(T5_config + "_" + canonical).iterrows():
    fig.add_trace(go.Scatter(x=x, y=row[score + "_scores"], mode='lines', name=row['data_usage_type']))
if score == "ndcg":
    for index, row in df_testing_retriver_reranker.iterrows():
        fig.add_trace(go.Scatter(x=x, y=row[score + "_scores"], mode='lines', name="golden standard"))
fig.update_layout(width=1400, height=700)
st.plotly_chart(fig, use_container_width=False, sharing="streamlit")

