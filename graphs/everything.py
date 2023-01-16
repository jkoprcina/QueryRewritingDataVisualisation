from show_data import *

def everything(st, manual):
    df = clean_data(pd.read_csv("data/everything.csv"))
    st.write(df)
    df_dictionary = split_df(df)

    st.write("all bleu and rouge scores on personal query rewriting methods table")
    df = df_dictionary["base_automatic"]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        data = st.radio("What data would you like to look at?", ('all', 'raw_vs_rewritten', 'similarity', 'summarizations'))
    with col2:
        T5_config = st.radio("What T5 config would you like to see?", ('base', 'personal'))
    with col3:
        canonical = st.radio("Would you like to see canonical or non-canonical?", ('canonical', 'automatic'))
    with col4:
        score = st.radio("Would you like to see the bleu or rouge scores?", ('bleu', 'rouge', 'ndcg'))

    st.subheader("You're looking at a graph showing the " + score + " score, ran on the T5 " + T5_config +
                 " configuration ran " + canonical + " for " + data)
    show_graph_options(st, data, T5_config, canonical, score, manual)