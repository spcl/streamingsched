"""
    Plots geomean of speedup as barplot
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import numpy as np
import scipy


def plot_speedup(dag: str, start: int, end: int, step: int, N: int, input_file: str, output_file: str, title: str):

    d = dict()
    medians = []
    whisker_low = []
    whisker_high = []
    outliers = []

    # build a new data frame with summarized and all results

    summary_results = pd.DataFrame({
        'PE': pd.Series(dtype='int'),
        'Type': pd.Series(dtype='str'),
        'Value': pd.Series(dtype='float')
    })
    results = pd.DataFrame({
        'PE': pd.Series(dtype='int'),
        'Type': pd.Series(dtype='str'),
        'Value': pd.Series(dtype='float')
    })
    summary_idx = 0
    idx = 0
    for i in range(start, end, step):

        if dag in {'resnet', 'mha'}:
            in_file = f"{input_file}/results_{dag}_P_{i}.csv"
        else:
            in_file = f"{input_file}/results_{dag}_N_{N}_P_{i}.csv"  # in_file = f"{input_file}/sim_error_resnet_P_{i}.csv"
        dataframe = pd.read_csv(in_file, error_bad_lines=False, encoding="UTF8")

        #Get the various data
        str_int_gang_speedup = dataframe["StrInt_Gang_Speedup"]
        non_stream_speedup = dataframe["NonStream_Speedup"]
        str_int_gang_speedup_geomean = scipy.stats.mstats.gmean(str_int_gang_speedup)
        non_stream_speedup_geomean = scipy.stats.mstats.gmean(non_stream_speedup)

        for index, value in dataframe.iterrows():
            results.loc[idx] = [i, "Streaming", value['StrInt_Gang_Speedup']]
            idx += 1
            results.loc[idx] = [i, "Non Streaming", value['NonStream_Speedup']]
            idx += 1

        summary_results.loc[summary_idx] = [i, "Streaming", str_int_gang_speedup_geomean]
        summary_idx += 1
        summary_results.loc[summary_idx] = [i, "Non Streaming", non_stream_speedup_geomean]
        summary_idx += 1

    print(summary_results)
    ### Seaborn figure setup
    sns.set(rc={'figure.figsize': (16, 10)})
    sns.set(font_scale=1.6)
    sns.set_style("whitegrid")

    # ax = sns.violinplot(data=df, linewidth=2)
    ax = sns.barplot(data=summary_results, x='PE', y='Value', hue='Type')
    # ax = sns.boxplot(data=results, x='PE', y='Value', hue='Type')
    ax.set_title(f"{dag.upper()}", fontsize=18)
    ax.set_ylabel("Speedup", fontsize=16)
    ax.set_xlabel("Num PEs", fontsize=16)

    # annotate median (or we can use df[<num_pe>].median)
    # and plot outliers
    # for i in range(len(medians)):
    #     ax.text(i + 0.2, medians[i], f"{medians[i]:.2f}")
    #     sns.scatterplot(x=i, y=outliers[i], marker='D', color='crimson', ax=ax)
    plt.tight_layout()
    if output_file:

        plt.savefig(output_file, dpi=200)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('start', type=int, help='file id start')
    parser.add_argument('end', type=int, help='file id end')
    parser.add_argument('step', type=int, help='file id step')
    parser.add_argument('dag', type=str, help='dag type: cholesky/fft/...')
    parser.add_argument('N', type=int, help='N of the Dag', nargs='?', default=1)

    parser.add_argument("-i", "--input", type=str, default=".")
    parser.add_argument("-o", "--output", type=str, default="plot.png")

    args = vars(parser.parse_args())
    input_file = args["input"]
    output_file = args["output"]

    start = args["start"]
    end = args["end"]
    step = args["step"]
    N = args["N"]
    dag = args["dag"]
    plot_speedup(dag, start, end, step, N, input_file, output_file, dag)
