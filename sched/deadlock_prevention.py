# Streaming-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.

import networkx as nx
from collections import defaultdict


def compute_buffer_space(dag: nx.DiGraph, spatial_blocks: list, schedule: dict, pseudo_source=0):
    """
    Computes the buffer space needed for each streaming edge in the DAG to avoid deadlocks

    :param dag: the DAG
    :param spatial_blocks: the spatial block lists
    :return: a dictionary containing for each streaming edge (src, dst) the corresponding buffer space
    """

    buffers_space = defaultdict(lambda: 1)

    # Pay attention to use the righ tasks schedule
    for comp in spatial_blocks:
        sb_chan_capacities = compute_buffer_space_of_block(dag, comp, schedule, pseudo_source=pseudo_source)
        # Python <3.9 compatibility
        buffers_space.update(sb_chan_capacities)

    return buffers_space


def compute_buffer_space_of_block(dag: nx.DiGraph, component, tasks_schedule, pseudo_source=0):
    """

    Compute buffer space in a given spatial block to avoid deadlocks by looking at:
    - first output times in undirected cycles
    - streaming intervals

    NOTE: max predecessor FO should represent the actual starting time of the node. We are currently using f_t
    """
    from math import ceil

    # TODO: remove
    # Build the subgraph of the streaming block
    subg = dag.subgraph(component)

    # loop over undirected cycles
    # TODO: understand what cycles we want to look at, if we want to remove edges after resolution, ...

    edges_buff_space = defaultdict(lambda: 1)
    num_cycles = 0

    from utils.graph import get_undirected_cycles

    # for source in source_nodes:
    for ucycle in get_undirected_cycles(subg, pseudo_source=pseudo_source):
        # print(ucycle)
        cycle_subg = subg.subgraph(ucycle)
        num_cycles += 1
        # Loop over the nodes in the cycle, and look if any node has in_degree > 1
        # (i.e.,  more than two predecessors in the same cycle)
        # TODO: is this really the case? maybe we should look at all predecessors (there could be something else impairing this)

        # print("Looking at ", cycle_subg.nodes)
        for node in cycle_subg.nodes:
            if cycle_subg.in_degree(node) > 1:
                # print("Candidate: ", node)
                # first get the max
                max_pred_fo = -1

                # Look at all predecessors (also the ones not in the cycle, also the ones that do no stream since they
                # will prevent the streaming one to send the data)

                for pred in subg.predecessors(node):
                    max_pred_fo = max(tasks_schedule[pred].f_t, max_pred_fo)
                    if 'stream' in dag.edges()[(pred, node)] and dag.edges()[(pred,
                                                                              node)]['stream']:  # The edge is streaming
                        max_pred_fo = max(tasks_schedule[pred].f_t, max_pred_fo)
                    else:
                        max_pred_fo = max(tasks_schedule[pred].end_t, max_pred_fo)

                # print(f"MAX PRED FO for {node}: {max_pred_fo}")

                for src, dst, data in cycle_subg.in_edges(node, data=True):
                    # print(
                    #     f"Src: {src}, fo src: {tasks_schedule[src].f_t}, max_pred_fo: {max_pred_fo}, streaming interval: {data['streaming_interval']}"
                    # )
                    buff_size = max(ceil((max_pred_fo - tasks_schedule[src].f_t) / data['streaming_interval']), 1)
                    if 'stream' in dag.edges()[(src, dst)] and dag.edges()[(src,
                                                                            dst)]['stream']:  # The edge is streaming
                        edges_buff_space[(src, dst)] = max(edges_buff_space[(src, dst)], buff_size)
    return edges_buff_space
