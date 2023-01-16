import plotly.graph_objects as go
import pandas as pd
from processing_data import *

def show_data(st, df_dictionary, query_string):
    df = df_dictionary.get(query_string)
    #df = df.sort_values(by='Sum of BLEU', ascending=False)
    #df = df.reset_index()
    df = df.drop(['bleu_scores', 'rouge_scores', 'ndcg_scores', 'ndcg_scores'], axis=1)
    st.write(df)
    return df


def show_graph_options(st, data, T5_config, canonical, score, manual):
    df = clean_data(pd.read_csv(get_route(data)))
    df_dictionary = split_df(df)

    x = [i for i in range(1, 13)]
    fig = go.Figure()
    for index, row in df_dictionary.get(T5_config + "_" + canonical).iterrows():
        fig.add_trace(go.Scatter(x=x, y=row[score + "_scores"], mode='lines', name=row['data_usage_type']))
    if score == "ndcg":
        fig.add_trace(go.Scatter(x=x, y=manual["ndcg"], mode='lines', name="golden standard"))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), width=1400, height=900)
    st.plotly_chart(fig, use_container_width=False, sharing="streamlit")


def show_comparison(st, first_string, second_string, first_df, second_df, manual):
    st.write("The average ndcg3 score for {0} is : {1:.4f} while {2} is : {3:.4f}          (while the golden standard is {4:.4f})".format(
        first_string, sum(list(first_df["ndcg3_sum"])) / len(list(first_df["ndcg3_sum"])),
        second_string, sum(list(second_df["ndcg3_sum"])) / len(list(second_df["ndcg3_sum"])),
        manual["ndcg_sum"]
    ))

    st.write("The average bleu score for {0}  is : {1:.4f} while {2} is : {3:.4f}".format(
        first_string, sum(list(first_df["bleu_sum"])) / len(list(first_df["bleu_sum"])),
        second_string, sum(list(second_df["bleu_sum"])) / len(list(second_df["bleu_sum"]))))

    st.write("The average rouge score for {0}  is : {1:.4f} while {2} is : {3:.4f}".format(
        first_string, sum(list(first_df["rouge_sum"])) / len(list(first_df["rouge_sum"])),
        second_string, sum(list(second_df["rouge_sum"])) / len(list(second_df["rouge_sum"]))))

def show_comparison_graph(st, dictionary, first_method, second_method, measure, manual,  x = [i for i in range(1, 13)]):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=dictionary[first_method + "_average_" + measure],  mode='lines',
                             name=first_method + "_average_" + measure + "_per_turn"))
    fig.add_trace(go.Scatter(x=x, y=dictionary[first_method + "_best_" + measure], mode='lines',
                             name=first_method + "_best_" + measure + "_per_turn"))
    fig.add_trace(go.Scatter(x=x, y=dictionary[second_method + "_average_" + measure], mode='lines',
                             name=second_method + "_average_" + measure + "_per_turn"))
    fig.add_trace(go.Scatter(x=x, y=dictionary[second_method + "_best_" + measure], mode='lines',
                             name=second_method + "_best_" + measure + "_per_turn"))
    if measure == "ndcg":
        fig.add_trace(go.Scatter(x=x, y=manual["ndcg"], mode='lines', name="golden standard"))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), width=1000, height=600)
    st.plotly_chart(fig, use_container_width=False, sharing="streamlit")