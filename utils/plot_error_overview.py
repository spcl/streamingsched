### Plots the sim error for the various experiments

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import numpy as np
import scipy
import math


def plot_sim_error(dag: str, start: int, end: int, step: int, N: int, input_file: str, title: str, ax):
    plot_type = "boxplot"  # boxplot/violinplot
    explicitely_plot_outlier = True
    top_k = 5  # How many outlier to plot

    d = dict()
    medians = []
    whisker_low = []
    whisker_high = []
    outliers = []

    results = pd.DataFrame({
        'PE': pd.Series(dtype='int'),
        'Type': pd.Series(dtype='str'),
        'Value': pd.Series(dtype='float')
    })
    idx = 0
    pal = sns.color_palette()

    for i in range(start, end, step):

        if dag in {'resnet', 'mha'}:
            in_file = f"{input_file}/sim_error_{dag}_P_{i}.csv"
        else:
            in_file = f"{input_file}/sim_error_{dag}_N_{N}_P_{i}.csv"

        dataframe = pd.read_csv(in_file, error_bad_lines=False, encoding="UTF8")

        sim_errors = dataframe[f"{i}"]
        sim_errors = sim_errors.apply(lambda x: x * 100)
        # print("Median: ", np.median(sim_errors))
        medians.append(np.median(sim_errors))
        d[i] = sim_errors
        q1, q3 = np.percentile(sim_errors, [25, 75])
        whisker_low.append(q1 - (q3 - q1) * 1.5)
        whisker_high.append(q3 + (q3 - q1) * 1.5)
        outl = sorted(sim_errors[(sim_errors > whisker_high[-1]) | (sim_errors < whisker_low[-1])])

        # Pick outliers (bottom-K and top-K)
        if len(outl) > top_k * 2:
            tmp = outl[0:top_k]
            tmp.append(outl[-top_k])

            outliers.append(tmp)
        else:
            outliers.append(outl)

        # Keep the top and bottom key

        print("Read: ", in_file)
        print(f"-------------- Stats {dag}, PE = {i} -------------- ")
        if len(outl) > top_k * 2:
            print("Outliers: ", outliers[-1])
        print("Whisker high: ", whisker_high[-1], " whisker low: ", whisker_low[-1])
        print("------------------------------------------------------")

        for value in sim_errors:
            results.loc[idx] = [i, "STR-SCH[SB-LTS]", value]
            idx += 1

        ##################################################
        # read the data for the no new block version
        ##################################################
        in_file = f"{input_file}_no_new_blocks/sim_error_{dag}_N_{N}_P_{i}.csv"
        dataframe = pd.read_csv(in_file, error_bad_lines=False, encoding="UTF8")
        sim_errors = dataframe[f"{i}"]
        sim_errors = sim_errors.apply(lambda x: x * 100)
        medians.append(np.median(sim_errors))
        q1, q3 = np.percentile(sim_errors, [25, 75])
        whisker_low.append(q1 - (q3 - q1) * 1.5)
        whisker_high.append(q3 + (q3 - q1) * 1.5)
        outl = sorted(sim_errors[(sim_errors > whisker_high[-1]) | (sim_errors < whisker_low[-1])])

        # Pick outliers (bottom-K and top-K)
        if len(outl) > top_k * 2:
            tmp = outl[0:top_k]
            tmp.append(outl[-top_k])

            outliers.append(tmp)
        else:
            outliers.append(outl)
        # continue previous numbering

        for value in sim_errors:
            results.loc[idx] = [i, "STR-SCH[SB-RLX]", value]
            idx += 1

    df = pd.DataFrame(d)
    print(df.head())

    # q1, q3 = np.percentile(df, [25, 75])
    # # https://stackoverflow.com/questions/66913456/how-to-mix-the-outliers-from-boxplot-or-not-with-violin-plot
    # whisker_low = q1 - (q3 - q1) * 1.5
    # whisker_high = q3 + (q3 - q1) * 1.5
    # print("Whisker high: ", whisker_high, " whisker low: ", whisker_low)
    # import pdb
    # pdb.set_trace()

    ### Seaborn figure setup
    sns.set(rc={'figure.figsize': (16, 10)})
    sns.set(font_scale=1.6)
    sns.set_style("whitegrid")

    if plot_type == "violin":
        subplt = sns.violinplot(data=df, linewidth=2, ax=ax)
    else:
        flierprops = dict(marker='o', markerfacecolor='None', markersize=10, markeredgecolor='black')
        # subplt = sns.boxplot(data=df, linewidth=2, showfliers=False, flierprops=flierprops, ax=ax, notch=True)
        # List of colors: https://matplotlib.org/stable/gallery/color/named_colors.html
        subplt = sns.boxplot(data=results, x='PE', y='Value', hue='Type', linewidth=2, showfliers=False, flierprops=flierprops, ax=ax,\
                                medianprops={ "alpha":1, "linewidth":3, "linestyle":"-"}),
    # ax.set(ylim=(-100, 40))
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Rel. Error (%)", fontsize=12)
    ax.set_xlabel("Num PEs", fontsize=12)

    plot_outliers = True

    if plot_outliers:
        lines = ax.get_lines()
        boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
        lines_per_box = int(len(lines) / len(boxes))
        i = 0
        for median in lines[4:len(lines):lines_per_box]:

            x, y = (data.mean() for data in median.get_data())
            # annotate median (or we can use df[<num_pe>].median)
            #ax.text(i + 0.2, medians[i], f"{medians[i]:.2f}")
            # and plot outliers
            # sns.scatterplot(x=i, y=outliers[i], marker='D', color='crimson', ax=ax)

            sns.scatterplot(x=x, y=outliers[i], marker="$\circ$", ec="face", color=pal[i % 2], s=80, ax=ax)
            i += 1
    ax.get_legend().remove()
    ax.legend([], [], frameon=False)

    return plt.subplot


if __name__ == '__main__':

    # Create the figure

    ### Seaborn figure setup
    sns.set(rc={'figure.figsize': (16, 5)})
    # sns.set(font_scale=1.6) # creates problem with the legend positioning
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(16, 5))

    ## Plot chain
    start = 2
    end = 8
    step = 2
    dag = "chain"
    N = 8
    title = f"Chain (#Tasks = {N})"
    # pay attention where this is called
    input_dir = "results_paper/second_iteration/small_dags_5it/chain_W_2048"
    chain_plot = plot_sim_error(dag, start, end + 1, step, N, input_dir, title, axes[0, 0])

    ## Plot FFT
    start = 32
    end = 128
    step = 32  # Note this was scaling even more than this
    dag = "fft"
    N = 32
    title = f"FFT (#Tasks = {int(2*N-1+N*math.log2(N))})"
    # pay attention where this is called
    input_dir = "results_paper/second_iteration/medium_dags_5it/fft_N_32_W_2048"
    plot_sim_error(dag, start, end + 1, step, N, input_dir, title, axes[0, 1])

    ## Plot Gaussian
    start = 32
    end = 128
    step = 32
    dag = "gaussian"
    N = 16
    title = f"Gaussian Elimination (#Tasks = {int((N*N+N-2)//2)})"
    # pay attention where this is called
    input_dir = "results_paper/second_iteration/medium_dags_5it/gaussian_N_16_W_2048"
    plot_sim_error(dag, start, end + 1, step, N, input_dir, title, axes[1, 0])

    ## Plot Cholesky
    start = 32
    end = 128
    step = 32
    N = 8
    dag = "cholesky"
    title = f"Cholesky Factorization (#Tasks = {int((N**3/6) + (N**2/2) + (N/3))})"
    # pay attention where this is called
    input_dir = "results_paper/second_iteration/medium_dags_5it/cholesky_N_8_W_2048"
    plot_sim_error(dag, start, end + 1, step, N, input_dir, title, axes[1, 1])

    # handles, labels = axes[1, 1].get_legend_handles_labels()
    # fig.legend(handles, labels, bbox_to_anchor=(0.65, 1.02), ncol=5, fontsize=14)

    # fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    # fig.legend(handles, labels, loc='center', ncol=2, bbox_to_anchor=(1, -0.1), bbox_transform=fig.transFigure)
    # axes[1, 0].legend(loc=(0.5, -1), ncol=2)

    # handles, labels = axes[1, 1].get_legend_handles_labels()
    # print(labels)
    # fig.legend(lines, labels, loc=(0.5, 0), ncol=5)
    # fig.legend(['Streaming', 'Non Streaming'], loc='lower right', bbox_to_anchor=(1, -0.1), ncol=2)
    #    bbox_transform=fig.transFigure)

    # General legend
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.65, 1.02), ncol=5, fontsize=12)

    plt.tight_layout()
    plt.show()  # Note it may seems that the legend is cut-out, but this is not the case at the end
    fig.savefig('plot_error.pdf', format='pdf', dpi=100)  # tight is needed to show the legend
