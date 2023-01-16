from show_data import *

def base_vs_personal(st, manual):
    df = pd.read_csv(get_route("all"))
    df_automatic = df[df['run_type'] == "automatic"]
    df_personal_automatic = df_automatic[df_automatic['T5_query_rewriter'] == "personal"]
    df_base_automatic = df_automatic[df_automatic['T5_query_rewriter'] == "base"]

    base_vs_personal_dict = {
        'personal_average_ndcg': get_average_ndcg_per_turn(df_personal_automatic),
        'personal_best_ndcg': get_best_ndcg(df_personal_automatic),
        'personal_average_bleu': get_average_bleu_per_turn(df_personal_automatic),
        'personal_best_bleu': get_best_bleu(df_personal_automatic),
        'personal_average_rouge': get_average_rouge_per_turn(df_personal_automatic),
        'personal_best_rouge': get_best_rouge(df_personal_automatic),
        'base_average_ndcg': get_average_ndcg_per_turn(df_base_automatic),
        'base_best_ndcg': get_best_ndcg(df_base_automatic),
        'base_average_bleu': get_average_bleu_per_turn(df_base_automatic),
        'base_best_bleu': get_best_bleu(df_base_automatic),
        'base_average_rouge': get_average_rouge_per_turn(df_base_automatic),
        'base_best_rouge': get_best_rouge(df_base_automatic)
    }

    st.subheader("Data summations and graph showing the comparison between the 2021. baseline TREC CAsT T5 model")
    st.subheader("and the T5 model we pretrained ourselves")

    show_comparison(st, "for our T5", "for the base T5", df_personal_automatic, df_base_automatic, manual)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        base_vs_personal_measure = st.radio("What data would you like to look at?", ('bleu', 'rouge', 'ndcg'), key="base_vs_personal")

    show_comparison_graph(st, base_vs_personal_dict, 'personal', 'base', base_vs_personal_measure, manual)