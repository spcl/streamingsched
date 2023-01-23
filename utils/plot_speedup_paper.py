### Plots the speedup for various experiments
"""
    Plots geomean of speedup as barplot
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import statistics
import argparse
import numpy as np
import scipy
import math
from matplotlib.patches import Patch


def add_median_labels(ax,
                      stream_pe_utilization,
                      stream2_pe_utilization,
                      non_stream_pe_utilization,
                      fmt='.2f',
                      percentage=True):
    # https://stackoverflow.com/questions/38649501/labeling-boxplot-in-seaborn-with-median-value
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))

    # print the pe utilization, quite ad hoc, this respect the numering
    i = 0
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation

        # print the median
        # value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        # print the pe utilization
        if i % 3 == 0:
            value = stream_pe_utilization[i // 3]
        elif i % 3 == 1:
            value = stream2_pe_utilization[i // 3]
        else:
            value = non_stream_pe_utilization[i // 3]
        if percentage:
            text = ax.text(x,
                           y * 1.2,
                           f'{int(value*100)}%',
                           ha='center',
                           va='center',
                           fontweight='bold',
                           fontsize=10,
                           color='white')
        else:
            text = ax.text(x,
                           y * 1.2,
                           f'{value:{fmt}}',
                           ha='center',
                           va='center',
                           fontweight='bold',
                           fontsize=10,
                           color='white')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])
        i = i + 1


def plot_speedup(dag: str,
                 start: int,
                 end: int,
                 step: int,
                 N: int,
                 input_file: str,
                 title: str,
                 ax,
                 use_boxplot: bool,
                 last=-1):
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
    stream2_pe_utilization = []  # no new block
    non_stream_pe_utilization = []

    read_no_block = True

    PEs = list(range(start, end, step))
    if last != -1:
        PEs += [last]

    for i in PEs:

        if dag in {'resnet', 'mha'}:
            in_file = f"{input_file}/results_{dag}_P_{i}.csv"
        else:
            in_file = f"{input_file}/results_{dag}_N_{N}_P_{i}.csv"  # in_file = f"{input_file}/sim_error_resnet_P_{i}.csv"
        dataframe = pd.read_csv(in_file, error_bad_lines=False, encoding="UTF8")

        #Get the various data
        str_int_gang_speedup = dataframe["StrInt_Gang_Speedup"]
        non_stream_speedup = dataframe["NonStream_Speedup"]
        str_int_gang_speedup_median = statistics.median(str_int_gang_speedup)
        non_stream_speedup_median = statistics.median(non_stream_speedup)

        stream_pe_utilization.append(str_int_gang_speedup_median / i)
        non_stream_pe_utilization.append(non_stream_speedup_median / i)
        # print("Stream PE utilization: ", stream_pe_utilization[-1], " non stream PE utilization ",
        #   non_stream_pe_utilization[-1])
        if use_boxplot:
            for index, value in dataframe.iterrows():
                results.loc[idx] = [i, "STR-SCH-1", value['StrInt_Gang_Speedup']]
                idx += 1
                results.loc[idx] = [i, "NSTR-SCH", value['NonStream_Speedup']]
                idx += 1

        # read also the new block version
        if read_no_block:
            in_file = f"{input_file}_no_new_blocks/results_{dag}_N_{N}_P_{i}.csv"
            dataframe = pd.read_csv(in_file, error_bad_lines=False, encoding="UTF8")

            str_int_gang_speedup = dataframe["StrInt_Gang_Speedup"]
            str_int_gang_speedup_median = statistics.median(str_int_gang_speedup)
            stream2_pe_utilization.append(str_int_gang_speedup_median / i)
            # continue previous numbering

            for index, value in dataframe.iterrows():
                results.loc[idx] = [i, "STR-SCH-2", value['StrInt_Gang_Speedup']]
                idx += 1

        summary_results.loc[summary_idx] = [i, "Pipelined", str_int_gang_speedup_median]
        summary_idx += 1
        summary_results.loc[summary_idx] = [i, "Buffered", non_stream_speedup_median]
        summary_idx += 1

    ### Seaborn figure setup
    # sns.set(rc={'figure.figsize': (16, 10)})
    # ax = sns.violinplot(data=df, linewidth=2)
    if use_boxplot:
        bplot = sns.boxplot(data=results,
                            x='PE',
                            y='Value',
                            hue='Type',
                            ax=ax,
                            showfliers=False,
                            hue_order=["STR-SCH-1", "STR-SCH-2", "NSTR-SCH"])
    else:
        bplot = sns.barplot(data=summary_results, x='PE', y='Value', hue='Type', ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Speedup", fontsize=12)
    ax.set_xlabel("Num PEs", fontsize=12)
    ax.get_legend().remove()

    # Annotate pe_utilization: we get first all the patches for streaming and the ones for non-streaming

    if use_boxplot:
        # for i, xtick in enumerate(bplot.get_xticks()):
        #     # print(i)
        #     bplot.text(xtick, 5, 1, horizontalalignment='center', size='x-small', color='b', weight='semibold')

        # TODO: use this as starting point
        add_median_labels(ax, stream_pe_utilization, stream2_pe_utilization, non_stream_pe_utilization)
        pass

    else:

        for i, p in enumerate(bplot.patches):

            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height() - (0.5 * bplot.get_ylim()[1]) / 3  # quite ad hoc placement

            if i < len(stream_pe_utilization):
                value = f"{stream_pe_utilization[i]:.2f}"
            else:
                value = f"{non_stream_pe_utilization[i-len(stream_pe_utilization)]:.2f}"
            print("Patch: ", i, " value: ", value, p.get_height())
            bplot.text(_x, _y, value, ha="center", va='bottom', color="white", weight="bold")

    # annotate median (or we can use df[<num_pe>].median)
    # and plot outliers
    # for i in range(len(medians)):
    #     ax.text(i + 0.2, medians[i], f"{medians[i]:.2f}")
    #     sns.scatterplot(x=i, y=outliers[i], marker='D', color='crimson', ax=ax)
    # plt.tight_layout()
    # if output_file:

    #     plt.savefig(output_file, dpi=200)
    # plt.show()
    return bplot


if __name__ == '__main__':

    # Create the figure

    ### Seaborn figure setup
    # Note: if you enlarge the preview given by plt.show, also the PDF will be enlarged
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
    last = -1
    title = f"FFT (#Tasks = {int(2*N-1+N*math.log2(N))})"
    # pay attention where this is called
    input_dir = "results_paper/second_iteration/medium_dags_5it/fft_N_32_W_2048"
    plot_speedup(dag, start, end + 1, step, N, input_dir, title, axes[0, 1], use_boxplot, last=last)

    ## Plot Gaussian
    start = 32
    end = 128
    step = 32
    dag = "gaussian"
    N = 16
    last = -1
    title = f"Gaussian Elimination (#Tasks = {int((N*N+N-2)//2)})"
    # pay attention where this is called
    input_dir = "results_paper/second_iteration/medium_dags_5it/gaussian_N_16_W_2048"
    plot_speedup(dag, start, end + 1, step, N, input_dir, title, axes[1, 0], use_boxplot, last=last)

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

    # Add entry to legend for PE utilization
    handles.append(Patch(facecolor='white', edgecolor='black'))
    labels.append("PE Utilization")
    fig.legend(handles, labels, bbox_to_anchor=(0.7, 1.02), ncol=4, fontsize=10, handletextpad=0.2, labelspacing=0.5)

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
    fig.savefig('plot_speedup.pdf', format='pdf', dpi=100, bbox_inches='tight')  # tight is needed to show the legend
