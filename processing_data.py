def clean_data(df):
    df = df.drop(['Id', 'Creation Time'], axis=1)
    bleu, rouge, ndcg = [], [], []
    for index, row in df.iterrows():
        bleu.append([row["b_s_2"], row["b_s_3"], row["b_s_4"], row["b_s_5"], row["b_s_6"], row["b_s_7"], row["b_s_8"],
                     row["b_s_9"], row["b_s_10"]])
        rouge.append([row["r_s_2 f_measure"], row["r_s_3 f_measure"], row["r_s_4 f_measure"], row["r_s_5 f_measure"],
                      row["r_s_6 f_measure"], row["r_s_7 f_measure"], row["r_s_8 f_measure"], row["r_s_9 f_measure"],
                      row["r_s_10 f_measure"]])
        ndcg.append([row["ndcg_2"], row["ndcg_3"], row["ndcg_4"], row["ndcg_5"], row["ndcg_6"], row["ndcg_7"],
                     row["ndcg_8"], row["ndcg_9"], row["ndcg_10"]])

    df = df.drop(['b_s_2', 'b_s_3', 'b_s_4', 'b_s_5', 'b_s_6', 'b_s_7', 'b_s_8', 'b_s_9', 'b_s_10', 'r_s_2 f_measure',
                  'r_s_3 f_measure', 'r_s_4 f_measure', 'r_s_5 f_measure', 'r_s_6 f_measure', 'r_s_7 f_measure',
                  'r_s_8 f_measure', 'r_s_9 f_measure', 'r_s_10 f_measure', 'ndcg_2', 'ndcg_3', 'ndcg_4', 'ndcg_5',
                  'ndcg_6', 'ndcg_7', 'ndcg_8', 'ndcg_9', 'ndcg_10'], axis=1)
    df = df.assign(bleu_scores=bleu)
    df = df.assign(rouge_scores=rouge)
    df = df.assign(ndcg_scores=ndcg)
    return df


def extract_data_for_line(df):
    blue_df = df.drop(['Sum of ROGUE', 'Sum of BLEU', 'r_s_2 f_measure', 'r_s_3 f_measure', 'r_s_4 f_measure',
                       'r_s_5 f_measure', 'r_s_6 f_measure', 'r_s_7 f_measure', 'r_s_8 f_measure', 'r_s_9 f_measure',
                       'r_s_10 f_measure'], axis=1)
    rouge_df = df.drop(['Sum of ROGUE', 'Sum of BLEU', 'b_s_2', 'b_s_3', 'b_s_4', 'b_s_5', 'b_s_6', 'b_s_7', 'b_s_8',
                        'b_s_9', 'b_s_10'], axis=1)
    return blue_df, rouge_df


def split_df(df):
    df_personal = df[df['T5_query_rewriter'] == "personal"]
    df_personal_testing_query_rewriting = df_personal[df_personal['run_type'] == "testing_query_rewriting"]
    df_personal_testing_everything = df_personal[df_personal['run_type'] == "testing_everything"]

    df_base = df[df['T5_query_rewriter'] == "base"]
    df_base_testing_query_rewriting = df_base[df_base['run_type'] == "testing_query_rewriting"]
    df_base_testing_everything = df_base[df_base['run_type'] == "testing_everything"]

    df_manual = df[df['run_type'] == "manual"]
    df_testing_retriever_reranker = df[df["run_type"] == "testing_retriever_reranker"]

    df_personal_testing_query_rewriting = df_personal_testing_query_rewriting.drop(['run_type', 'T5_query_rewriter'], axis=1)
    df_personal_testing_everything = df_personal_testing_everything.drop(['run_type', 'T5_query_rewriter'], axis=1)
    df_base_testing_query_rewriting = df_base_testing_query_rewriting.drop(['run_type', 'T5_query_rewriter'], axis=1)
    df_base_testing_everything = df_base_testing_everything.drop(['run_type', 'T5_query_rewriter'], axis=1)

    return (df_manual, df_testing_retriever_reranker, df_personal_testing_query_rewriting,
            df_personal_testing_everything, df_base_testing_query_rewriting, df_base_testing_everything)

