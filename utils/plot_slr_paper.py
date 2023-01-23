### Plots the speedup for various experiments
"""
    Plots geomean of speedup as barplot
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

import argparse
import numpy as np
import scipy
import math


def add_median_labels(ax, fmt='.1f'):
    # https://stackoverflow.com/questions/38649501/labeling-boxplot-in-seaborn-with-median-value
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center', fontweight='bold', color='white')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])


def plot_speedup(dag: str, start: int, end: int, step: int, N: int, input_file: str, title: str, ax, use_boxplot: bool):
    '''
    :param dag: the dag name: chol/chain/fft/gaussian
    :param start, end, step: the range of number of PEs
    :param N: parameter of dag (e.g., chain with N stages)
    :param ax: where to plot

    '''

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

    stream_pe_utilization = []
    non_stream_pe_utilization = []

    for i in range(start, end, step):

        if dag in {'resnet', 'mha'}:
            in_file = f"{input_file}/results_{dag}_P_{i}.csv"
        else:
            in_file = f"{input_file}/results_{dag}_N_{N}_P_{i}.csv"  # in_file = f"{input_file}/sim_error_resnet_P_{i}.csv"
        dataframe = pd.read_csv(in_file, error_bad_lines=False, encoding="UTF8")

        #Get the various data

        # print("Stream PE utilization: ", stream_pe_utilization[-1], " non stream PE utilization ",
        #   non_stream_pe_utilization[-1])
        for index, value in dataframe.iterrows():
            results.loc[idx] = [i, "STR-SCH[SB-LTS]", value['StreamingSLR']]
            idx += 1

        # read the data for the no new block
        in_file = f"{input_file}_no_new_blocks/results_{dag}_N_{N}_P_{i}.csv"
        dataframe = pd.read_csv(in_file, error_bad_lines=False, encoding="UTF8")
        # continue previous numbering

        for index, value in dataframe.iterrows():
            results.loc[idx] = [i, "STR-SCH[SB-RLX]", value['StreamingSLR']]
            idx += 1

            # results.loc[idx] = [i, "Non Streaming", value['NonStreamingSLR']]
            # idx += 1

    # ax = sns.violinplot(data=df, linewidth=2)
    bplot = sns.boxplot(data=results, x='PE', y='Value', hue='Type', ax=ax, showfliers=False)

    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Streaming SLR", fontsize=12)
    ax.set_xlabel("Num PEs", fontsize=12)
    ax.get_legend().remove()

    # Annotate pe_utilization: we get first all the patches for streaming and the ones for non-streaming

    # for i, xtick in enumerate(bplot.get_xticks()):
    #     # print(i)
    #     bplot.text(xtick, 5, 1, horizontalalignment='center', size='x-small', color='b', weight='semibold')

    # add_median_labels(ax)

    return bplot


if __name__ == '__main__':

    # Create the figure

    ### Seaborn figure setup
    sns.set(rc={'figure.figsize': (16, 5)})
    # sns.set(font_scale=1.6) # creates problem with the legend positioning
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(16, 5))
    use_boxplot = True

    ## Plot chain
    start = 2
    end = 8
    step = 2
    dag = "chain"
    N = 8
    title = f"Chain (#Tasks = {N})"
    # pay attention where this is called
    input_dir = "results_paper/second_iteration/small_dags_5it/chain_W_2048"
    chain_plot = plot_speedup(dag, start, end + 1, step, N, input_dir, title, axes[0, 0], use_boxplot)

    ## Plot FFT
    start = 32
    end = 128
    step = 32  # Note this was scaling even more than this
    dag = "fft"
    N = 32
    title = f"FFT (#Tasks = {int(2*N-1+N*math.log2(N))})"
    # pay attention where this is called
    input_dir = "results_paper/second_iteration/medium_dags_5it/fft_N_32_W_2048"

    plot_speedup(dag, start, end + 1, step, N, input_dir, title, axes[0, 1], use_boxplot)

    ## Plot Gaussian
    start = 32
    end = 128
    step = 32
    dag = "gaussian"
    N = 16
    title = f"Gaussian Elimination (#Tasks = {int((N*N+N-2)//2)})"
    # pay attention where this is called
    input_dir = "results_paper/second_iteration/medium_dags_5it/gaussian_N_16_W_2048"
    plot_speedup(dag, start, end + 1, step, N, input_dir, title, axes[1, 0], use_boxplot)

    ## Plot Cholesky
    start = 32
    end = 128
    step = 32
    N = 8
    dag = "cholesky"
    title = f"Cholesky Factorization (#Tasks = {int((N**3/6) + (N**2/2) + (N/3))})"
    # pay attention where this is called
    input_dir = "results_paper/second_iteration/medium_dags_5it/cholesky_N_8_W_2048"

    plot_speedup(dag, start, end + 1, step, N, input_dir, title, axes[1, 1], use_boxplot)

    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.65, 1.02), ncol=5, fontsize=12)

    # fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    # fig.legend(handles, labels, loc='center', ncol=2, bbox_to_anchor=(1, -0.1), bbox_transform=fig.transFigure)
    # axes[1, 0].legend(loc=(0.5, -1), ncol=2)

    # handles, labels = axes[1, 1].get_legend_handles_labels()
    # print(labels)
    # fig.legend(lines, labels, loc=(0.5, 0), ncol=5)
    # fig.legend(['Streaming', 'Non Streaming'], loc='lower right', bbox_to_anchor=(1, -0.1), ncol=2)
    #    bbox_transform=fig.transFigure)

    plt.tight_layout()
    plt.show()  # Note it may seems that the legend is cut-out, but this is not the case at the end
    fig.savefig('plot_slr.pdf', format='pdf', dpi=100, bbox_inches='tight')  # tight is needed to show the legend
