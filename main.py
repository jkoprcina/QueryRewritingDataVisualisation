import streamlit as st
from show_data import *
from graphs.everything import everything
from graphs.base_vs_personal import base_vs_personal
from graphs.automatic_vs_canonical import automatic_vs_canonical
from graphs.raw_vs_rewritten import raw_vs_rewritten
from graphs.metric_comparison import metric_comparison
st.set_page_config(layout="wide")

df = clean_data_2(pd.read_csv("data/small.csv"))
df_dictionary = split_df(df)
BM25_MonoBERT_df = df_dictionary.get("base_automatic")
BM25_MonoBERT_df = df.drop(['bleu_scores', 'rouge_scores', 'bleu_sum', 'rouge_sum', 'ndcg_scores', 'run_type'], axis=1)
st.write("BM25 -> LuceneSearch BM25 by pyserini")
st.write("MonoBERT ->  MonoBERT from pygaggle which uses 'castorini/monobert-large-msmarco'")
st.write(BM25_MonoBERT_df)

df = pd.read_csv("data/manual.csv")
df = df.drop(['Id', 'Creation Time'], axis=1)
st.write(df)

for index, row in df.iterrows():
    ndcg = [
        row["ndcg_1"] * 100, row["ndcg_2"] * 100, row["ndcg_3"] * 100, row["ndcg_4"] * 100,
        row["ndcg_5"] * 100,  row["ndcg_6"] * 100, row["ndcg_7"] * 100, row["ndcg_8"] * 100,
        row["ndcg_9"] * 100, row["ndcg_10"] * 100, row["ndcg_11"] * 100, row["ndcg_12"] * 100
    ]
    ndcg3_sum = row["ndcg3_sum"]
manual = {"ndcg_sum": ndcg3_sum, "ndcg": ndcg}

metric_comparison(st)
everything(st, manual)
base_vs_personal(st, manual)
automatic_vs_canonical(st, manual)
raw_vs_rewritten(st, manual)