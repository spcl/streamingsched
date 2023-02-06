# Streaming-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Utility functions to visualize a graph
'''
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import numpy as np
import matplotlib.font_manager as font_manager
from pyvis.network import Network


def visualize_dag(G, node_labels_attribute=None):
    '''
    Visualize DAG
    :param G:
    :param node_labels_attribute: if specified identifies the node attribute to use as node label
    :return:
    '''

    node_options = {
        "node_size": 1000,
        "linewidths": 2,
        "node_color": "white",  # fill color
        "edgecolors": "black"  # line color
    }
    edge_options = {"edge_color": "black", "width": 3, "arrows": True, "arrowsize": 20}
    streaming_edge_options = {"edge_color": "red", "width": 3, "arrows": True, "arrowsize": 20, "style": "dashed"}
    # pos = nx.spring_layout(G, seed=50)  # Seed layout for reproducibility
    # pos = nx.planar_layout(G)
    # pos = nx.random_layout(G)
    # Use graphviz layout for a more hierarchical plot
    pos = graphviz_layout(G, prog="dot")

    # get streaming and non-streaming edges
    e_streaming = [(u, v) for (u, v, d) in G.edges(data=True) if "stream" in d and d["stream"]]
    e_non_streaming = [(u, v) for (u, v, d) in G.edges(data=True) if not "stream" in d or not d["stream"]]

    edge_labels = dict([((n1, n2), f"{G[n1][n2]['weight']}") for n1, n2 in G.edges])
    #############
    # Drawing
    #############
    # nodes
    nx.draw_networkx_nodes(G, pos, **node_options, node_shape="o")
    # # edges
    nx.draw_networkx_edges(G, pos, edgelist=e_streaming, **streaming_edge_options)
    nx.draw_networkx_edges(G, pos, edgelist=e_non_streaming, **edge_options)
    #
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    if node_labels_attribute:
        labels = nx.get_node_attributes(G, node_labels_attribute)
        nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif", labels=labels)
    else:
        nx.draw_networkx_labels(G, pos, font_size=14, font_family="sans-serif")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig('/tmp/graph.png')
    plt.show()


def visualize_dag_pyvis(G):
    '''
    Visualize the DAG using the interactive PyVis
    :param G:
    :return:
    '''
    #create vis network
    net = Network(height="720px", width="1080px")
    #load the the network
    net.from_nx(G)
    for edge in net.edges:
        edge['label'] = f"{edge['label']}"

    # net.show_buttons()
    #show
    net.show("dag.html")
    return

    g = Network(height="720px", width="1080px")

    # for node, data in G.nodes(data=True):
    #     g.add_node(node, label=data['label'])
    g.add_node(1)
    g.add_node(0)
    g.add_node(2)
    g.add_node(6)
    g.add_node(28)
    g.add_node(9)

    for src in nx.algorithms.topological_sort(G):
        for _, dst, data in G.out_edges(src, data=True):
            # is_streaming = is_edge_streaming_explicit(dag, src, dst)
            # print(f"dag.add_edge({src}, {dst}, weight = {data['weight']}, stream = {is_streaming})")

            g.add_edge(src, dst, label=f"{data['weight']}")
        break
    g.show_buttons(filter_=["physics", "edges"])

    g.show('nx.html')


def show_schedule_gantt_chart(pes_schedule: dict, task_labels=None, pe_labels: list = None, save_to=None):
    """
        Displays a Gantt chart generated using Matplotlib for the PEs schedule
        :param task_labels: mapping node_id->label
        :param pe_labels: labels to use for PE (must be compliant with pes_schedule)
    """
    if pes_schedule == None:
        pass
    pes = list(pes_schedule.keys())

    color_choices = ['red', 'blue', 'green', 'cyan', 'magenta']

    if pe_labels is not None:
        # build color array: one color per Processing Element type
        pe_colors = []
        used_colors = {}
        color_idx = 0
        for idx, p in enumerate(pe_labels):
            if p not in used_colors:
                used_colors[p] = color_idx
                color_idx = color_idx + 1
            pe_colors.append(used_colors[p])

    ilen = len(pes)
    pos = np.arange(0.5, ilen * 0.5 + 0.5, 0.5)
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(111)
    for idx, pe in enumerate(pes):
        for job in pes_schedule[pe]:
            job_duration = job.end_t - job.start_t
            if job_duration > 0:  # Do not show jobs with duration = 0
                ax.barh((idx * 0.5) + 0.5,
                        job_duration,
                        left=job.start_t,
                        height=0.3,
                        align='center',
                        edgecolor='black',
                        color='white',
                        alpha=0.95)

                label = str(job.task) if task_labels is None else task_labels[job.task]
                fontsize = 16 if task_labels is None else 10
                label_color = color_choices[((job.task) // 100) %
                                            5] if pe_labels is None else color_choices[pe_colors[job.pe]]
                text_align_offset = 0.4 * job_duration if task_labels is None else 0
                ax.text(text_align_offset + job.start_t, (idx * 0.5) + 0.5 - 0.03125,
                        label,
                        color=label_color,
                        fontweight='bold',
                        fontsize=fontsize,
                        alpha=0.75)
    if pe_labels is None:
        pe_labels = []
        for idx in range(0, len(pes)):
            pe_labels.append(idx)
    locsy, labelsy = plt.yticks(pos, pe_labels)
    plt.ylabel('PE', fontsize=16)
    plt.xlabel('Time (clock cycles)', fontsize=16)
    plt.setp(labelsy, fontsize=14)
    ax.set_ylim(ymin=-0.1, ymax=ilen * 0.5 + 0.5)
    ax.set_xlim(xmin=-5)
    ax.grid(color='g', linestyle=':', alpha=0.5)

    font = font_manager.FontProperties(size='small')
    if save_to:
        plt.savefig(save_to)
    plt.show()
