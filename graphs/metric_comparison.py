from show_data import *
import numpy as np

def metric_comparison(st):
    df = pd.read_csv(get_route("all"))
    df = df[df['T5_query_rewriter'] == "base"]
    df = df[df['run_type'] == "automatic"]
    df = df.drop(['Id', 'Creation Time', 'T5_query_rewriter', 'run_type'], axis=1)

    metrics_dict = {
        'average_ndcg': get_average_ndcg_per_turn(df),
        'average_bleu': get_average_bleu_per_turn(df),
        'average_rouge': get_average_rouge_per_turn(df)
    }

    st.subheader("Graph showing the comparison between different metrics at different turns")

    st.write("The correlation between rouge and bleu is :" +
             str(np.corrcoef(metrics_dict['average_bleu'], metrics_dict['average_rouge'])[0, 1]))

    st.write("The correlation between bleu and ndcg is :" +
             str(np.corrcoef(metrics_dict['average_bleu'], metrics_dict['average_ndcg'])[0, 1]))

    st.write("The correlation between rouge and ndcg is :" +
             str(np.corrcoef(metrics_dict['average_rouge'], metrics_dict['average_ndcg'])[0, 1]))

    x = [i for i in range(1, 13)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=metrics_dict['average_ndcg'], mode='lines', name="average_ndcg_per_turn"))
    fig.add_trace(go.Scatter(x=x, y=metrics_dict['average_bleu'], mode='lines', name="average_bleu_per_turn"))
    fig.add_trace(go.Scatter(x=x, y=metrics_dict['average_rouge'], mode='lines', name="average_rouge_per_turn"))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), width=1000, height=600)
    st.plotly_chart(fig, use_container_width=False, sharing="streamlit")