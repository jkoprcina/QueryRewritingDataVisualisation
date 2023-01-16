from show_data import *


def automatic_vs_canonical(st, manual):
    df = pd.read_csv(get_route("automatic_vs_canonical"))
    df_base = df[df['T5_query_rewriter'] == "base"]
    df_base = df_base.drop(['Id', 'Creation Time', 'T5_query_rewriter'], axis=1)

    df_base_automatic = df_base[df_base['run_type'] == "automatic"]
    df_base_canonical = df_base[df_base['run_type'] == "canonical"]

    automatic_vs_canonical_dict = {
        'automatic_average_ndcg': get_average_ndcg_per_turn(df_base_automatic),
        'automatic_best_ndcg': get_best_ndcg(df_base_automatic),
        'automatic_average_bleu': get_average_bleu_per_turn(df_base_automatic),
        'automatic_best_bleu': get_best_bleu(df_base_automatic),
        'automatic_average_rouge': get_average_rouge_per_turn(df_base_automatic),
        'automatic_best_rouge': get_best_rouge(df_base_automatic),
        'canonical_average_ndcg': get_average_ndcg_per_turn(df_base_canonical),
        'canonical_best_ndcg': get_best_ndcg(df_base_canonical),
        'canonical_average_bleu': get_average_bleu_per_turn(df_base_canonical),
        'canonical_best_bleu': get_best_bleu(df_base_canonical),
        'canonical_average_rouge': get_average_rouge_per_turn(df_base_canonical),
        'canonical_best_rouge': get_best_rouge(df_base_canonical)
    }

    st.subheader("Data summations and graph showing the comparison between automatic and canonical runs")
    show_comparison(st, "for automatic runs", "for canonical runs", df_base_automatic, df_base_canonical, manual)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        automatic_vs_canonical_measure = st.radio("What data would you like to look at?", ('bleu', 'rouge', 'ndcg'), key="automatic_vs_canonical")

    show_comparison_graph(st, automatic_vs_canonical_dict, 'automatic', 'canonical', automatic_vs_canonical_measure, manual)