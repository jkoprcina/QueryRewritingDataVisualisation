def clean_data(df):
    df = df.drop(['Id', 'Creation Time'], axis=1)
    bleu, rouge, ndcg = [], [], []
    for index, row in df.iterrows():
        bleu.append([
            row["bleu_1"], row["bleu_2"], row["bleu_3"], row["bleu_4"],  row["bleu_5"],  row["bleu_6"],
            row["bleu_7"], row["bleu_8"], row["bleu_9"], row["bleu_10"], row["bleu_11"], row["bleu_12"]])
        rouge.append([
            row["rouge_1"], row["rouge_2"], row["rouge_3"], row["rouge_4"],  row["rouge_5"],  row["rouge_6"],
            row["rouge_7"], row["rouge_8"], row["rouge_9"], row["rouge_10"], row["rouge_11"], row["rouge_12"]])
        ndcg.append([
            row["ndcg_1"] * 100, row["ndcg_2"] * 100, row["ndcg_3"] * 100, row["ndcg_4"] * 100,
            row["ndcg_5"] * 100,  row["ndcg_6"] * 100, row["ndcg_7"] * 100, row["ndcg_8"] * 100,
            row["ndcg_9"] * 100, row["ndcg_10"] * 100, row["ndcg_11"] * 100, row["ndcg_12"] * 100])

    df = df.drop(['bleu_1',  'bleu_2',  'bleu_3',  'bleu_4',   'bleu_5',   'bleu_6',
                  'bleu_7',  'bleu_8',  'bleu_9',  'bleu_10',  'bleu_11',  'bleu_12',
                  'rouge_1', 'rouge_2', 'rouge_3', 'rouge_4',  'rouge_5',  'rouge_6',
                  'rouge_7', 'rouge_8', 'rouge_9', 'rouge_10', 'rouge_11', 'rouge_12',
                  'ndcg_1',  'ndcg_2',  'ndcg_3',  'ndcg_4',   'ndcg_5',   'ndcg_6',
                  'ndcg_7',  'ndcg_8',  'ndcg_9',  'ndcg_10',  'ndcg_11',  'ndcg_12',], axis=1)
    df = df.assign(bleu_scores=bleu)
    df = df.assign(rouge_scores=rouge)
    df = df.assign(ndcg_scores=ndcg)
    return df


def clean_data_2(df):
    df = df.drop(['Id', 'Creation Time'], axis=1)
    bleu, rouge, ndcg, ndcg_formatted = [], [], [], []
    for index, row in df.iterrows():
        bleu.append([row["bleu_2"], row["bleu_3"], row["bleu_4"], row["bleu_5"], row["bleu_6"],
                     row["bleu_7"], row["bleu_8"], row["bleu_9"], row["bleu_10"]])
        rouge.append([row["rouge_2"], row["rouge_3"], row["rouge_4"], row["rouge_5"], row["rouge_6"],
                      row["rouge_7"], row["rouge_8"], row["rouge_9"], row["rouge_10"]])
        ndcg.append([row["ndcg_1"] * 100, row["ndcg_2"] * 100, row["ndcg_3"] * 100, row["ndcg_4"] * 100,
                     row["ndcg_5"] * 100, row["ndcg_6"] * 100, row["ndcg_7"] * 100, row["ndcg_8"] * 100,
                     row["ndcg_9"] * 100, row["ndcg_10"] * 100])
        ndcg_formatted.append([
            format(row["ndcg_1"] * 100, ".2f"), format(row["ndcg_2"] * 100, ".2f"), format(row["ndcg_3"] * 100, ".2f"),
            format(row["ndcg_4"] * 100, ".2f"), format(row["ndcg_5"] * 100, ".2f"), format(row["ndcg_6"] * 100, ".2f"),
            format(row["ndcg_7"] * 100, ".2f"), format(row["ndcg_8"] * 100, ".2f"), format(row["ndcg_9"] * 100, ".2f"),
            format(row["ndcg_10"] * 100, ".2f")])

    df = df.drop(['bleu_2', 'bleu_3', 'bleu_4', 'bleu_5', 'bleu_6', 'bleu_7', 'bleu_8', 'bleu_9', 'bleu_10',
                  'rouge_2', 'rouge_3', 'rouge_4', 'rouge_5', 'rouge_6', 'rouge_7', 'rouge_8', 'rouge_9', 'rouge_10',
                  'ndcg_1', 'ndcg_2', 'ndcg_3', 'ndcg_4', 'ndcg_5', 'ndcg_6', 'ndcg_7', 'ndcg_8', 'ndcg_9', 'ndcg_10'], axis=1)
    df = df.assign(bleu_scores=bleu)
    df = df.assign(rouge_scores=rouge)
    df = df.assign(ndcg_scores=ndcg)
    df = df.assign(ndcg_formatted=ndcg_formatted)
    return df


def extract_data_for_line(df):
    blue_df = df.drop(['Sum of ROGUE', 'Sum of BLEU', 'r_s_2 f_measure', 'r_s_3 f_measure', 'r_s_4 f_measure',
                       'r_s_5 f_measure', 'r_s_6 f_measure', 'r_s_7 f_measure', 'r_s_8 f_measure', 'r_s_9 f_measure',
                       'r_s_10 f_measure'], axis=1)
    rouge_df = df.drop(['Sum of ROGUE', 'Sum of BLEU', 'b_s_2', 'b_s_3', 'b_s_4', 'b_s_5', 'b_s_6', 'b_s_7', 'b_s_8',
                        'b_s_9', 'b_s_10'], axis=1)
    return blue_df, rouge_df


def extract_single_run_data(df):
    df_manual = df[df['run_type'] == "manual"]
    df_testing_retriever_reranker = df[df["run_type"] == "testing_retriever_reranker"]
    return df_manual, df_testing_retriever_reranker


def split_df(df):
    df_personal = df[df['T5_query_rewriter'] == "personal"]
    df_personal_canonical = df_personal[df_personal['run_type'] == "canonical"]
    df_personal_automatic = df_personal[df_personal['run_type'] == "automatic"]

    df_base = df[df['T5_query_rewriter'] == "base"]
    df_base_canonical = df_base[df_base['run_type'] == "canonical"]
    df_base_automatic = df_base[df_base['run_type'] == "automatic"]

    df_personal_canonical = df_personal_canonical.drop(['run_type', 'T5_query_rewriter'], axis=1)
    df_personal_automatic = df_personal_automatic.drop(['run_type', 'T5_query_rewriter'], axis=1)
    df_base_canonical = df_base_canonical.drop(['run_type', 'T5_query_rewriter'], axis=1)
    df_base_automatic = df_base_automatic.drop(['run_type', 'T5_query_rewriter'], axis=1)

    df_dictionary = {
        'base_canonical': df_base_canonical,
        'base_automatic': df_base_automatic,
        'personal_canonical': df_personal_canonical,
        'personal_automatic': df_personal_automatic
    }

    return df_dictionary


def get_route(data):
    try:
        if data == "all":
            route = "data/everything.csv"
        elif data == "raw_vs_rewritten_vs_golden":
            route = "data/raw_rewritten_golden.csv"
        elif data == "similarity":
            route = "data/similarity.csv"
        elif data == "summarizations":
            route = "data/summarizations.csv"
        elif data == "automatic_vs_canonical":
            route = "data/automatic_canonical.csv"
    except:
        raise Exception("Option for route doesn't exist")
    return route

def get_average_ndcg_per_turn(df):
    return [
        sum(list(df["ndcg_1"])) / len(list(df["ndcg_1"])) * 100,
        sum(list(df["ndcg_2"])) / len(list(df["ndcg_2"])) * 100,
        sum(list(df["ndcg_3"])) / len(list(df["ndcg_3"])) * 100,
        sum(list(df["ndcg_4"])) / len(list(df["ndcg_4"])) * 100,
        sum(list(df["ndcg_5"])) / len(list(df["ndcg_5"])) * 100,
        sum(list(df["ndcg_6"])) / len(list(df["ndcg_6"])) * 100,
        sum(list(df["ndcg_7"])) / len(list(df["ndcg_7"])) * 100,
        sum(list(df["ndcg_8"])) / len(list(df["ndcg_8"])) * 100,
        sum(list(df["ndcg_9"])) / len(list(df["ndcg_9"])) * 100,
        sum(list(df["ndcg_10"])) / len(list(df["ndcg_10"])) * 100,
        sum(list(df["ndcg_11"])) / len(list(df["ndcg_11"])) * 100,
        sum(list(df["ndcg_12"])) / len(list(df["ndcg_12"])) * 100
    ]
def get_average_bleu_per_turn(df):
    return [
        sum(list(df["bleu_1"])) / len(list(df["bleu_1"])),
        sum(list(df["bleu_2"])) / len(list(df["bleu_2"])),
        sum(list(df["bleu_3"])) / len(list(df["bleu_3"])),
        sum(list(df["bleu_4"])) / len(list(df["bleu_4"])),
        sum(list(df["bleu_5"])) / len(list(df["bleu_5"])),
        sum(list(df["bleu_6"])) / len(list(df["bleu_6"])),
        sum(list(df["bleu_7"])) / len(list(df["bleu_7"])),
        sum(list(df["bleu_8"])) / len(list(df["bleu_8"])),
        sum(list(df["bleu_9"])) / len(list(df["bleu_9"])),
        sum(list(df["bleu_10"])) / len(list(df["bleu_10"])),
        sum(list(df["bleu_11"])) / len(list(df["bleu_11"])),
        sum(list(df["bleu_12"])) / len(list(df["bleu_12"]))
    ]
def get_average_rouge_per_turn(df):
    return [
        sum(list(df["rouge_1"])) / len(list(df["rouge_1"])),
        sum(list(df["rouge_2"])) / len(list(df["rouge_2"])),
        sum(list(df["rouge_3"])) / len(list(df["rouge_3"])),
        sum(list(df["rouge_4"])) / len(list(df["rouge_4"])),
        sum(list(df["rouge_5"])) / len(list(df["rouge_5"])),
        sum(list(df["rouge_6"])) / len(list(df["rouge_6"])),
        sum(list(df["rouge_7"])) / len(list(df["rouge_7"])),
        sum(list(df["rouge_8"])) / len(list(df["rouge_8"])),
        sum(list(df["rouge_9"])) / len(list(df["rouge_9"])),
        sum(list(df["rouge_10"])) / len(list(df["rouge_10"])),
        sum(list(df["rouge_11"])) / len(list(df["rouge_11"])),
        sum(list(df["rouge_12"])) / len(list(df["rouge_12"]))
    ]
def get_best_ndcg(df):
    x = df.nlargest(1, 'ndcg3_sum')
    for index, row in x.iterrows():
        return [
            row["ndcg_1"] * 100, row["ndcg_2"] * 100, row["ndcg_3"] * 100, row["ndcg_4"] * 100,
            row["ndcg_5"] * 100, row["ndcg_6"] * 100, row["ndcg_7"] * 100, row["ndcg_8"] * 100,
            row["ndcg_9"] * 100, row["ndcg_10"] * 100, row["ndcg_11"] * 100, row["ndcg_12"] * 100
        ]
def get_best_bleu(df):
    x = df.nlargest(1, 'bleu_sum')
    for index, row in x.iterrows():
        return [
            row["bleu_1"], row["bleu_2"], row["bleu_3"], row["bleu_4"],  row["bleu_5"],  row["bleu_6"],
            row["bleu_7"], row["bleu_8"], row["bleu_9"], row["bleu_10"], row["bleu_11"], row["bleu_12"]
        ]
def get_best_rouge(df):
    x = df.nlargest(1, 'rouge_sum')
    for index, row in x.iterrows():
        return [
            row["rouge_1"], row["rouge_2"], row["rouge_3"], row["rouge_4"],  row["rouge_5"],  row["rouge_6"],
            row["rouge_7"], row["rouge_8"], row["rouge_9"], row["rouge_10"], row["rouge_11"], row["rouge_12"]
        ]
