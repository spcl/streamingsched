import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import numpy as np


def plot_sim_error(dag: str, start: int, end: int, step: int, N: int, input_file: str, output_file: str, title: str):
    plot_type = "boxplot"  # boxplot/violinplot
    explicitely_plot_outlier = True
    top_k = 5  # How many outlier to plot

    d = dict()
    medians = []
    whisker_low = []
    whisker_high = []
    outliers = []
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
    print(outliers)
    df = pd.DataFrame(d)
    print(df.head())

    q1, q3 = np.percentile(df, [25, 75])
    # https://stackoverflow.com/questions/66913456/how-to-mix-the-outliers-from-boxplot-or-not-with-violin-plot
    whisker_low = q1 - (q3 - q1) * 1.5
    whisker_high = q3 + (q3 - q1) * 1.5
    # import pdb
    # pdb.set_trace()

    ### Seaborn figure setup
    sns.set(rc={'figure.figsize': (16, 10)})
    sns.set(font_scale=1.6)
    sns.set_style("whitegrid")

    if plot_type == "violin":
        ax = sns.violinplot(data=df, linewidth=2)
    # outliers = df[(data > whisker_high) | (data < whisker_low)]
    # sns.scatterplot(x=outliers, y=0, marker='D', color='crimson', ax=a)
    # ax = sns.boxenplot(data=df, linewidth=2)
    else:
        flierprops = dict(marker='o', markerfacecolor='None', markersize=10, markeredgecolor='black')
        # https://stackoverflow.com/questions/59955080/change-width-of-median-line-of-boxenplot-in-seaborn

        # List of colors: https://matplotlib.org/stable/gallery/color/named_colors.html
        ax = sns.boxplot(data=df, linewidth=2, showfliers=False, flierprops=flierprops, \
                                medianprops={"color":"darkturquoise", "alpha":1, "linewidth":5, "linestyle":"-"},
                         )
    # ax.set(ylim=(-100, 40))
    ax.set_title(f"{dag.upper()}", fontsize=18)
    ax.set_ylabel("Err %", fontsize=16)
    ax.set_xlabel("Num PEs", fontsize=16)

    plot_outliers = True

    if plot_outliers:
        for i in range(len(medians)):
            # annotate median (or we can use df[<num_pe>].median)
            #ax.text(i + 0.2, medians[i], f"{medians[i]:.2f}")
            # and plot outliers
            # sns.scatterplot(x=i, y=outliers[i], marker='D', color='crimson', ax=ax)
            sns.scatterplot(x=i, y=outliers[i], marker="$\circ$", ec="face", color='black', s=80, ax=ax)
    #######################
    # Set size and save

    fig = plt.gcf()
    fig.set_size_inches(16, 5)
    plt.tight_layout()
    if output_file:

        plt.savefig(output_file, dpi=200)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # positional arguments
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
    plot_sim_error(dag, start, end, step, N, input_file, output_file, dag)
