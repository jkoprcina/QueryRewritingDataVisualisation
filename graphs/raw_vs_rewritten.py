from show_data import *

def raw_vs_rewritten(st, manual):
    df = pd.read_csv(get_route("raw_vs_rewritten_vs_golden"))
    df = df[df["run_type"] == "automatic"]
    df = df[df["T5_query_rewriter"] == "base"]
    raw_vs_rewritten_vs_golden_dict = {}
    df = df.drop(['Id', 'Creation Time', 'T5_query_rewriter', 'run_type'], axis=1)

    df_raw = df[df['data_usage_type'].str.contains("raw")]
    raw_vs_rewritten_vs_golden_dict['raw_average_ndcg'] = get_average_ndcg_per_turn(df_raw)
    raw_vs_rewritten_vs_golden_dict['raw_best_ndcg'] = get_best_ndcg(df_raw)
    raw_vs_rewritten_vs_golden_dict['raw_average_bleu'] = get_average_bleu_per_turn(df_raw)
    raw_vs_rewritten_vs_golden_dict['raw_best_bleu'] = get_best_bleu(df_raw)
    raw_vs_rewritten_vs_golden_dict['raw_average_rouge'] = get_average_rouge_per_turn(df_raw)
    raw_vs_rewritten_vs_golden_dict['raw_best_rouge'] = get_best_rouge(df_raw)

    df_rewritten = df[df['data_usage_type'].str.contains("rewritten")]
    raw_vs_rewritten_vs_golden_dict['rewritten_average_ndcg'] = get_average_ndcg_per_turn(df_rewritten)
    raw_vs_rewritten_vs_golden_dict['rewritten_best_ndcg'] = get_best_ndcg(df_rewritten)
    raw_vs_rewritten_vs_golden_dict['rewritten_average_bleu'] = get_average_bleu_per_turn(df_rewritten)
    raw_vs_rewritten_vs_golden_dict['rewritten_best_bleu'] = get_best_bleu(df_rewritten)
    raw_vs_rewritten_vs_golden_dict['rewritten_average_rouge'] = get_average_rouge_per_turn(df_rewritten)
    raw_vs_rewritten_vs_golden_dict['rewritten_best_rouge'] = get_best_rouge(df_rewritten)

    df_golden = df[df['data_usage_type'].str.contains("gold")]
    st.write(df_golden)
    raw_vs_rewritten_vs_golden_dict['golden_average_ndcg'] = get_average_ndcg_per_turn(df_golden)
    raw_vs_rewritten_vs_golden_dict['golden_best_ndcg'] = get_best_ndcg(df_golden)
    raw_vs_rewritten_vs_golden_dict['golden_average_bleu'] = get_average_bleu_per_turn(df_golden)
    raw_vs_rewritten_vs_golden_dict['golden_best_bleu'] = get_best_bleu(df_golden)
    raw_vs_rewritten_vs_golden_dict['golden_average_rouge'] = get_average_rouge_per_turn(df_golden)
    raw_vs_rewritten_vs_golden_dict['golden_best_rouge'] = get_best_rouge(df_golden)

    st.subheader("Data summations and graph showing the comparison between runs using raw queries, rewritten queries and golden queries")

    st.write("The average ndcg3 score for raw queries is : {0:.4f}, for rewritten queries is : {1:.4f}, while for golden queries is : {2:.4f}".format(
        sum(list(df_raw["ndcg3_sum"])) / len(list(df_raw["ndcg3_sum"])),
        sum(list(df_rewritten["ndcg3_sum"])) / len(list(df_rewritten["ndcg3_sum"])),
        sum(list(df_golden["ndcg3_sum"])) / len(list(df_golden["ndcg3_sum"])),
        manual["ndcg_sum"]
    ))

    st.write("The average bleu score for raw queries is : {0:.4f}, for rewritten queries is : {1:.4f}, while for golden queries is : {2:.4f}".format(
        sum(list(df_raw["bleu_sum"])) / len(list(df_raw["bleu_sum"])),
        sum(list(df_rewritten["bleu_sum"])) / len(list(df_rewritten["bleu_sum"])),
        sum(list(df_golden["bleu_sum"])) / len(list(df_golden["bleu_sum"]))))

    st.write("The average rouge score for raw queries is : {0:.4f}, for rewritten queries is : {1:.4f}, while for golden queries is : {2:.4f}".format(
        sum(list(df_raw["rouge_sum"])) / len(list(df_raw["rouge_sum"])),
        sum(list(df_rewritten["rouge_sum"])) / len(list(df_rewritten["rouge_sum"])),
        sum(list(df_golden["rouge_sum"])) / len(list(df_golden["rouge_sum"]))))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        measure = st.radio("What data would you like to look at?", ('bleu', 'rouge', 'ndcg'), key="raw_vs_rewritten")

    x = [i for i in range(1, 13)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=raw_vs_rewritten_vs_golden_dict["raw_average_" + measure], mode='lines',
                             name="raw_average_" + measure + "_per_turn"))
    fig.add_trace(go.Scatter(x=x, y=raw_vs_rewritten_vs_golden_dict["raw_best_" + measure], mode='lines',
                             name="raw_best_" + measure + "_per_turn"))

    fig.add_trace(go.Scatter(x=x, y=raw_vs_rewritten_vs_golden_dict["rewritten_average_" + measure], mode='lines',
                             name="rewritten_average_" + measure + "_per_turn"))
    fig.add_trace(go.Scatter(x=x, y=raw_vs_rewritten_vs_golden_dict["rewritten_best_" + measure], mode='lines',
                             name="rewritten_best_" + measure + "_per_turn"))

    fig.add_trace(go.Scatter(x=x, y=raw_vs_rewritten_vs_golden_dict["golden_average_" + measure], mode='lines',
                             name="golden_average_" + measure + "_per_turn"))
    fig.add_trace(go.Scatter(x=x, y=raw_vs_rewritten_vs_golden_dict["golden_best_" + measure], mode='lines',
                             name="golden_best_" + measure + "_per_turn"))

    if measure == "ndcg":
        fig.add_trace(go.Scatter(x=x, y=manual["ndcg"], mode='lines', name="golden standard"))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), width=1000, height=600)
    st.plotly_chart(fig, use_container_width=False, sharing="streamlit")
