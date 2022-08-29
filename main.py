import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from processing_data import clean_data, split_df

st.set_page_config(layout="wide")
df = pd.read_csv("data/Query-Rewriting.csv")
df = clean_data(df)
df_personal_canonical, df_personal_not_canonical, df_base_canonical, df_base_not_canonical = split_df(df)
df_dictionary = {'base_canonical': df_base_canonical, 'base_non_canonical': df_base_not_canonical,
                 'personal_canonical': df_personal_canonical, 'personal_non_canonical': df_personal_not_canonical}


col1, col2, col3 = st.columns(3)
with col1:
    T5_config = st.radio("What T5 config would you like to see?", ('base', 'personal'))
with col2:
    canonical = st.radio("Would you like to see canonical or non-canonical?", ('canonical', 'non_canonical'))
with col3:
    score = st.radio("Would you like to see the bleu or rouge scores?", ('bleu', 'rouge'))

st.subheader("You're looking at a graph showing the " + score + " score, ran on the T5 " + T5_config +
             " configuration ran " + canonical)

x = [i for i in range(2, 11)]
fig = go.Figure()
for index, row in df_dictionary.get(T5_config + "_" + canonical).iterrows():
    fig.add_trace(go.Scatter(x=x, y=row[score + "_scores"], mode='lines', name=row['sent_extra']))
fig.update_layout(width=1400, height=700)
st.plotly_chart(fig, use_container_width=False, sharing="streamlit")

