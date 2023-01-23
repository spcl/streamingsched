# S-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
Heuristics for spatial partitioning
"""

import networkx as nx
import numpy as np
from collections import deque, namedtuple
import logging
import matplotlib.pyplot as plt
from queue import PriorityQueue
from collections import deque
from dataclasses import dataclass
import math
from queue import PriorityQueue, Queue
import sympy as sym
import copy
from networkx.algorithms import all_simple_paths, all_simple_edge_paths
from utils.graph import get_sink_nodes, get_source_nodes, is_edge_streaming, get_pseudo_sink
from sched import StreamingScheduler
from utils.visualize import visualize_dag
from fractions import Fraction


def critical_path(G: nx.DiGraph, reverse=True, topo_order=None):
    '''
    Returns the critical path of the given graph
    Borrowed from nx longest path implementation, but modified to take into account the weight of the node
    rather than the edge weight
    :return:
    '''
    if not G:
        return []

    if topo_order is None:
        topo_order = nx.topological_sort(G)

    dist = {}  # stores {v : (length, u)}
    for v in topo_order:
        # TODO: use the cost of v
        v_weight = G.nodes()[v]['weight']
        us = [(dist[u][0] + v_weight, u) for u, data in G.pred[v].items()]

        # Use the best predecessor if there is one and its distance is
        # non-negative, otherwise terminate.
        maxu = max(us, key=lambda x: x[0]) if us else (0, v)
        dist[v] = maxu if maxu[0] >= 0 else (0, v)

    u = None
    v = max(dist, key=lambda x: dist[x][0])
    path = []
    while u != v:
        path.append(v)
        u = v
        v = dist[v][1]

    if reverse:
        path.reverse()

    # remove from the path pseudo root
    if 0 in path:
        path.remove(0)

    return path


def spatial_block_partitioning(G: nx.DiGraph,
                               num_pes,
                               pseudo_root,
                               pseudo_sink_node,
                               buffer_nodes={},
                               remove_edges=[],
                               create_new_blocks=True):
    """
    Spatial Block Partitioning: partitions the graph in components of at most P nodes that will be co-scheduled.
    As each spatial block is co-scheduled, all edges between computational tasks of the same spatial block can be streaming edges.

    The heuristic comes in two variants. 

    - if create_new_blocks is set to True,  we add a node to a spatial block if its produced data volume is less than the data
         volume produced by the block's source(s) from which it depends (if any). We continue adding to the same spatial block until 
         such node does exist or the block is full. Otherwise, we create a new spatial block, and we start filling it.
    - if the argument create_new_blocks is set to False, if no other candidate is available, a node can be added to the current 
        spatial block even if it is producing more data than the block's source(s). 
        In this case, all spatial blocks (except the last one) contain num_pes tasks.

    :param: _G_ the graph
    :param: _num_pes_ the number of PEs
    :param: _pseudo_root_ and _pseudo_sink_nodes_
    :param: _buffer_nodes_ the set of buffer nodes
    :param: _remove_edges_ a list of edges to remove (optional)
    :param: _create_new_blocks_: flag to choose the heuristic to apply
    :return: the list of streaming paths and the spatial block partitions (list of lists)
    """

    from sched.streaming_sched import StreamingScheduler
    # keep track of the streaming paths enabled so far
    streaming_paths = []

    # Convenience data structure: keep track of the collapsed nodes
    # list of streaming components (set)
    streaming_components = list()
    streaming_components.append(list())
    streaming_components_sources = list()
    streaming_components_sources.append(list())

    new_dag = copy.deepcopy(G)

    # analyze the dag to compute the streaming intervals
    scheduler = StreamingScheduler(new_dag, num_pes=num_pes, buffer_nodes=buffer_nodes)

    # for u, v, data in new_dag.edges(data=True):
    #     print(u, v, " -> ", data['streaming_interval'])
    # Now create streaming blocks by considering the actual running time of each node
    # This is given by its work and incident streaming intervals

    work = [0] * new_dag.number_of_nodes()
    for n in new_dag.nodes():
        if n == pseudo_root or n == pseudo_sink_node:
            work[n] = 0
        else:
            work[n] = scheduler._compute_execution_time_isolation(n)

    # Compute the depth of the nodes to break ties
    depths = [0] * new_dag.number_of_nodes()
    for node in nx.algorithms.bfs_tree(new_dag, pseudo_root):
        max_depth = 0
        for p in new_dag.predecessors(node):
            max_depth = max(max_depth, depths[p])
        depths[node] = max_depth + 1

    # save output data for all the nodes
    # TODO: how to deal with the pseudo source?

    consumed_data = [0] * new_dag.number_of_nodes()
    produced_data = [0] * new_dag.number_of_nodes()

    for n in new_dag.nodes:
        if new_dag.out_degree(n) > 0:
            produced_data[n] = list(new_dag.out_edges(n, data=True))[0][2]['weight']
        if new_dag.in_degree(n) > 0:
            consumed_data[n] = list(new_dag.in_edges(n, data=True))[0][2]['weight']

    ################ Remove edges if indicated
    for edge in remove_edges:
        new_dag.remove_edge(edge[0], edge[1])

    # TODO: we should use the produced data or the work?
    # Work seems better, since sources of a SB may have work higher than produced data
    produced_data = work

    consider_upsampling_ratio = False
    max_in_volume = -1
    buffer_nodes_in_current_block = 0

    sources_descendants = {}  # dict: node -> descendants for quick lookup
    while (new_dag.number_of_nodes() > 0):

        # get the source nodes
        sources = get_source_nodes(new_dag)

        first_candidate = None
        added = False

        # we can add a node to the current streaming block if the work produced
        # is less than the work of the (streaming block) source from which it depends
        # If this is not the case, we can consider to create a new block (create_new_blocks=True) or not

        candidate_for_new_block = -1
        candidate_upsampling = -1
        added_something = False
        candidate = -1
        add_to_sources = False
        max_slowdown = -1  # for upsampling nodes
        # print("Source: ", sources)
        for s in sources:

            # skip pseudo nodes
            if s == pseudo_root or s == pseudo_sink_node:
                new_dag.remove_node(s)
                break

            # does this node depends from a source of this block?
            dependant = False
            for bs in streaming_components_sources[-1]:
                # check if this node is descendant from a source in this block
                # (since we remove the nodes from the dag, we look up in the original one. Since we are adding
                # the node s, this means that all predecessors have been already added)
                if s in sources_descendants[bs]:
                    dependant = True
                    break

            # if this depends from another source, we need to check the data being produced
            if dependant:

                if consider_upsampling_ratio and max_in_volume > 0:
                    # Not used
                    # look at the input streaming interval (use theo 3.1, considering the input volume)
                    input_streaming_interval = Fraction(max_in_volume, consumed_data[s])
                    production_ratio = produced_data[s] / consumed_data[s]
                    # print(
                    #     f"Node {s}, production_ratio: {production_ratio}, input streaming interval: {input_streaming_interval}, max_in_volume: {max_in_volume}"
                    # )
                    if input_streaming_interval < production_ratio:
                        # if we add this node, it will be impaired by a slow input streaming interval.
                        # Add it to a new block
                        # print(
                        #     f"-----Adding {s} will slow down all the other.Max slow down: {max_slowdown}, this slow down:{input_streaming_interval/production_ratio}"
                        # )
                        if max_slowdown == -1 or production_ratio / input_streaming_interval < max_slowdown:
                            candidate_upsampling = s
                            max_slowdown = production_ratio / input_streaming_interval
                        continue

                if produced_data[s] <= produced_data[bs] or s in buffer_nodes:
                    # add the nodes with less data than the source (or buffer nodes)

                    if consider_upsampling_ratio and max_in_volume > 0:
                        # look at the input streaming interval (use theo 3.1, considering the input volume)
                        input_streaming_interval = Fraction(max_in_volume, consumed_data[s])
                        production_ratio = produced_data[s] / consumed_data[s]

                        if input_streaming_interval > production_ratio:
                            # if we add this node, it will be impaired by a slow input streaming interval.
                            # Add it to a new block
                            if max_slowdown == -1 or input_streaming_interval / production_ratio < max_slowdown:
                                candidate_upsampling = s
                                max_slowdown = input_streaming_interval / production_ratio
                            continue

                    if candidate == -1 or produced_data[s] >= produced_data[
                            candidate]:  # among all descendant that satisfy the condition, choose the one with max work
                        # we can add it to this streaming block

                        # print(
                        #     f"Descendent candidate selected: {s} (pd: {produced_data[s]} depends from {bs} (pd: {produced_data[bs]}),     previous candidate was {candidate} (pd: {produced_data[candidate] if candidate!= -1 else 0})"
                        # )
                        candidate = s
                else:
                    # it needs to go on another SB

                    if create_new_blocks:
                        if candidate_for_new_block == -1 or produced_data[s] > produced_data[candidate_for_new_block]:
                            candidate_for_new_block = s
                    else:
                        if candidate_for_new_block == -1 or produced_data[s] < produced_data[candidate_for_new_block]:
                            candidate_for_new_block = s
            else:
                # This is a new streaming block source: we can safely add it to this streaming block and to the sources of the block
                candidate = s
                add_to_sources = True
                break

        if candidate != -1:

            # we've found a node that is either a source or has less work than the source
            added_something = True
            streaming_components[-1].append(candidate)
            if candidate in buffer_nodes:
                buffer_nodes_in_current_block += 1
            # print("\tAdding: ", candidate)
            # if needed added to sources
            if add_to_sources:
                streaming_components_sources[-1].append(candidate)
                sources_descendants[candidate] = nx.descendants(G, candidate)

            max_in_volume = max(max_in_volume, consumed_data[candidate])
            max_in_volume = max(max_in_volume, produced_data[candidate])

            # if this streaming block is complete add a new one
            if len(streaming_components[-1]) - buffer_nodes_in_current_block >= num_pes:
                #create a new streaming component
                streaming_components.append(list())
                streaming_components_sources.append(list())
                max_in_volume = 0
                buffer_nodes_in_current_block = 0

            new_dag.remove_node(candidate)
        elif s != pseudo_root and s != pseudo_sink_node:

            # if we didn't add something, the only option is to create a new streaming block

            if consider_upsampling_ratio and candidate_upsampling != -1 and candidate_for_new_block == -1:
                # print("Adding an upsampling node anyway: ", candidate_upsampling)
                candidate_for_new_block = candidate_upsampling

            assert candidate_for_new_block != -1 and new_dag.number_of_nodes() > 0
            if create_new_blocks:
                # create a new block
                # print("\tCreating a new block with node: ", candidate_for_new_block)
                streaming_components.append(list())
                streaming_components_sources.append(list())
                buffer_nodes_in_current_block = 0
                streaming_components[-1].append(candidate_for_new_block)
                streaming_components_sources[-1].append(candidate_for_new_block)
                sources_descendants[candidate_for_new_block] = nx.descendants(G, candidate_for_new_block)
                max_in_volume = consumed_data[candidate_for_new_block]
                if candidate_for_new_block in buffer_nodes:
                    buffer_nodes_in_current_block += 1

            else:
                #add to the old one
                streaming_components[-1].append(candidate_for_new_block)
                max_in_volume = max(max_in_volume, consumed_data[candidate_for_new_block])
                max_in_volume = max(max_in_volume, produced_data[candidate_for_new_block])
                if candidate_for_new_block in buffer_nodes:
                    buffer_nodes_in_current_block += 1
                if len(streaming_components[-1]) - buffer_nodes_in_current_block >= num_pes:
                    #create a new streaming component
                    streaming_components.append(list())
                    streaming_components_sources.append(list())
                    max_in_volume = 0
                    buffer_nodes_in_current_block = 0

            new_dag.remove_node(candidate_for_new_block)

    # get all the edges between the nodes in the streaming components
    # We don't stream from buffer nodes
    for component in streaming_components:
        for src in component:
            if src in buffer_nodes:
                continue
            for dst in component:
                if G.has_edge(src, dst):
                    # print("Adding ", src, dst)
                    streaming_paths.append((src, dst))
    # add the pseudo_root node to the first component, and pseudo_sink to the last
    if pseudo_root not in streaming_components[0]:
        streaming_components[0].insert(0, pseudo_root)
    if pseudo_sink_node in G.nodes() and pseudo_sink_node not in streaming_components[-1]:
        streaming_components[-1].append(pseudo_sink_node)

    return streaming_paths, streaming_components


def spatial_block_partitioning_running_time(G: nx.DiGraph, num_pes, pseudo_root, pseudo_sink_node):
    """

    Another spatial block partitioning heuristic (does not take into account the presence of buffer nodes)

    Uses streaming intervals to compute running time. Pick the ready node with maximum running time.
    Do not add it if it is an upsampling node, with an upsampling ratio lower than the
    input edge streaming interval
    """
    from sched.streaming_sched import StreamingScheduler
    # keep track of the streaming paths enabled so far
    streaming_paths = []

    # Convenience data structure: keep track of the collapsed nodes
    # list of streaming components (set)
    streaming_components = list()
    streaming_components.append(list())

    new_dag = copy.deepcopy(G)

    # analyze the dag to compute the streaming intervals
    scheduler = StreamingScheduler(new_dag, num_pes=num_pes)

    # Compute streaming intervals assuming every edge is streaming
    scheduler.streaming_interval_analysis()
    # for u, v, data in new_dag.edges(data=True):
    #     print(u, v, " -> ", data['streaming_interval'])
    # Now create streaming blocks by considering the actual running time of each node
    # This is given by its work and incident streaming intervals

    # compute the execution time  of each node
    exec_time = [0] * new_dag.number_of_nodes()
    # reuse the scheduler TODO optimize it
    for n in new_dag.nodes():
        if n == pseudo_root or n == pseudo_sink_node:
            continue
        else:
            work = scheduler._compute_execution_time_isolation(n)
            exec_time[n] = scheduler._compute_average_execution_in_schedule(n, 0, [], use_streaming_intervals=True)
            # print("Exec time for ", n, exec_time[n], work)

    # Compute the depth of the nodes to break ties
    depths = [0] * new_dag.number_of_nodes()
    for node in nx.algorithms.bfs_tree(new_dag, pseudo_root):
        max_depth = 0
        for p in new_dag.predecessors(node):
            max_depth = max(max_depth, depths[p])
        depths[node] = max_depth + 1

    while (new_dag.number_of_nodes() > 0):

        # get the source nodes
        sources = get_source_nodes(new_dag)

        first_candidate = None
        added = False

        while len(sources) > 0:

            # get the node with max running time
            candidate = -1
            max_etime = -1
            depth = 0
            for n in sources:
                if exec_time[n] == max_etime:
                    # keep the node with highest depth
                    if depths[n] < depth:
                        candidate = n
                        depth = depths[n]
                elif exec_time[n] > max_etime:
                    max_etime = exec_time[n]
                    candidate = n
                    depth = depths[n]

            if first_candidate is None:
                first_candidate = candidate

            # skip root or pseudo sink node
            if candidate == pseudo_root or candidate == pseudo_sink_node:
                new_dag.remove_node(candidate)
                break

            # get the upsampling ratio of the candidate
            if G.in_degree(candidate) > 0 and G.out_degree(candidate) > 0:
                #check in the original dag
                # import pdb
                # pdb.set_trace()

                input_data = list(G.in_edges(candidate, data=True))[0][2]['weight']
                output_data = list(G.out_edges(candidate, data=True))[0][2]['weight']
                upsampling_ratio = max(output_data / input_data, 1)
                # print("candidate : ", candidate, " ur: ", upsampling_ratio)
            else:
                upsampling_ratio = 1

            if len(streaming_components[-1]) >= num_pes:
                #create a new streaming component
                streaming_components.append(list())

            # check that the streaming interval of the edge arriving to
            # the candidate is less than its upsampling ratio
            # TODO optimize this
            if len(streaming_components[-1]) > 0 and upsampling_ratio > 1:
                subg = nx.subgraph(G, streaming_components[-1] + [candidate]).copy()

                # let it have a single source and a single sink

                source_nodes = [node for node in subg.nodes if subg.in_degree(node) == 0]

                # TODO: this must be done better, we have to put a meaningful weight for this node
                id = max(list(subg.nodes()))
                if len(source_nodes) > 1:
                    id += 1
                    for sn in source_nodes:
                        # if same_weight_for_all_ops:
                        #     resulting_dag.add_edge(id, sn, weight=same_weight_for_all_ops)
                        # else:
                        # get the output weight and use that, so that it is like an elwise

                        output_data = list(G.out_edges(sn, data=True))[0][2]['weight']
                        subg.add_edge(id, sn, weight=output_data)

                exit_nodes = [node for node in subg.nodes if subg.out_degree(node) == 0]
                if len(exit_nodes) > 1:
                    id += 1
                    subg.add_node(id, pseudo=True)
                    for en in exit_nodes:
                        input_data = list(G.in_edges(en, data=True))[0][2]['weight']
                        subg.add_edge(en, id, weight=input_data)

                for src, dst, data in subg.edges(data=True):
                    data['stream'] = True
                subg_sched = StreamingScheduler(subg, num_pes=1)
                # TODO: fix steady state analysis generic because here it has some problem
                subg_sched.streaming_interval_analysis()

                # if candidate == 3:
                #     import pdb
                #     pdb.set_trace()
                # get the input edge if any
                if subg.in_degree(candidate) > 0:
                    input_edge_str_interval = list(subg.in_edges(candidate, data=True))[0][2]['streaming_interval']
                    if input_edge_str_interval > upsampling_ratio:

                        # This is not a good candidate: let's try to find another source and see if that can be added to the current
                        # streaming component

                        # remove from the source list
                        sources.remove(candidate)
                        continue  # Try with another source

                        # print(
                        #     f"It is not convenient to add {candidate} to {streaming_components[-1]} ({input_edge_str_interval}> {upsampling_ratio}) "
                        # )
                        # TODO maybe we can continue adding other sources?
                        #create a new streaming component
                        streaming_components.append(list())
                    # else:
                    #     print(
                    #         f"It is convenient to add {candidate} to {streaming_components[-1]} ({input_edge_str_interval} < {upsampling_ratio}) "
                    #     )

            # If we arrive here, then the candidate is ok, we can add it to the streaming component
            streaming_components[-1].append(candidate)
            added = True

            # remove the node from the dag
            new_dag.remove_node(candidate)
            break

        if candidate != pseudo_root and candidate != pseudo_sink_node and not added:
            # use the first one that you have found, create a new streaming component
            streaming_components.append(list())
            streaming_components[-1].append(candidate)
            # remove the node from the dag
            new_dag.remove_node(candidate)

    # get all the edges between the nodes in the streaming components
    for component in streaming_components:
        for src in component:
            for dst in component:
                if G.has_edge(src, dst):
                    # print("Adding ", src, dst)
                    streaming_paths.append((src, dst))
    # print("Streaming paths: ", streaming_paths)
    # add the pseudo_root node to the first component, and pseudo_sink to the last
    if pseudo_root not in streaming_components[0]:
        streaming_components[0].insert(0, pseudo_root)
    if pseudo_sink_node in G.nodes() and pseudo_sink_node not in streaming_components[-1]:
        streaming_components[-1].append(pseudo_sink_node)
    return streaming_paths, streaming_components


def spatial_block_partitioning_max_work(G: nx.DiGraph, num_pes):
    '''
    Detects streaming blocks (hence streaming edges) only by considering the work of each node

    Can work for elementwise and downsampler graphs.

    :param dag:
    :return: streaming paths, streaming blocks (internally ordered according to the DAG dependencies)
    '''

    # TODO: we don't want to modify the original DAG
    new_dag = copy.deepcopy(G)

    # keep track of the streaming paths enabled so far
    streaming_paths = []

    # Convenience data structure: keep track of the collapsed nodes
    # list of streaming components (set)
    streaming_components = list()
    streaming_components.append(list())
    pseudo_root = get_source_nodes(G)[0]
    pseudo_sink_node = get_pseudo_sink(G)

    # compute the work of each node
    work = [0] * new_dag.number_of_nodes()
    streaming_scheduler = StreamingScheduler(G, num_pes=num_pes)
    for n in new_dag.nodes():
        if n == pseudo_root or n == pseudo_sink_node:
            continue
        else:
            work[n] = streaming_scheduler._compute_execution_time_isolation(n)

    # Compute the depth of the nodes to break ties
    depths = [0] * new_dag.number_of_nodes()
    for node in nx.algorithms.bfs_tree(new_dag, pseudo_root):
        max_depth = 0
        for p in new_dag.predecessors(node):
            max_depth = max(max_depth, depths[p])
        depths[node] = max_depth + 1

    while (new_dag.number_of_nodes() > 0):

        # get the source nodes
        sources = get_source_nodes(new_dag)

        # get the node with max work
        candidate = -1
        max_work = -1
        depth = 0
        for n in sources:
            if work[n] == max_work:
                # keep the node with highest depth
                if depths[n] < depth:
                    candidate = n
                    depth = depths[n]
            elif work[n] > max_work:
                max_work = work[n]
                candidate = n
                depth = depths[n]

        if candidate != pseudo_root and candidate != pseudo_sink_node:
            # add the candidate to the current streaming component
            if len(streaming_components[-1]) >= num_pes:
                #create a new streaming component
                streaming_components.append(list())

            streaming_components[-1].append(candidate)

        # remove the node from the dag
        new_dag.remove_node(candidate)

    # get all the edges between the nodes in the streaming components
    for component in streaming_components:
        for src in component:
            for dst in component:
                if G.has_edge(src, dst):
                    streaming_paths.append((src, dst))
    # print("Streaming paths: ", streaming_paths)
    # add the pseudo_root node to the first component, and pseudo_sink to the last
    if pseudo_root not in streaming_components[0]:
        streaming_components[0].insert(0, pseudo_root)
    if pseudo_sink_node in G.nodes() and pseudo_sink_node not in streaming_components[-1]:
        streaming_components[-1].append(pseudo_sink_node)

    return streaming_paths, streaming_components
