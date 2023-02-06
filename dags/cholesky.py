# Streaming-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Builds the DAG for a block-Cholesky decomposition
    of an NxN tiles matrix
    (left-looking)
'''

import numpy as np
from utils.metrics import *
from utils.graph import *
import random
import math


def build_cholesky_dag(N=4, W=16, random_nodes=True, max_output_volume=-1):
    '''
    Builds a random Cholesky Canonical task graph
    :param N: number of tiles
    :param: W: the starting weight
    :param random_nodes: a boolean flag that indicates whether the DAG must have node with randomly generated production rates.
        Otherwise, all tasks are elementwise tasks
    :return: the dag, the root and the sink nodes
    '''

    dag = nx.DiGraph()
    root = 0
    dag.add_node(0, weight=0)  # pseudo root

    last_acted_on = {}  # Given a tile (pair of coordinates) returns the last task that operated on it
    current_node = 1
    for k in range(N):

        for n in range(k):
            # A[k][k] = SYRK(A[k][n], A[k][k])

            dag.add_node(current_node, label="SYRK")
            if (k, n) in last_acted_on:
                dag.add_edge(last_acted_on[(k, n)], current_node, weight=W)

            if (k, k) in last_acted_on:
                dag.add_edge(last_acted_on[(k, k)], current_node, weight=W)

            last_acted_on[(k, k)] = current_node
            current_node += 1

        # A[k][k] = POTRF(A[k][k])
        dag.add_node(current_node, label="POTRF")

        if k == 0:  # first task
            dag.add_edge(0, current_node, weight=W)

        if (k, k) in last_acted_on:
            dag.add_edge(last_acted_on[(k, k)], current_node, weight=W)

        last_acted_on[(k, k)] = current_node
        current_node += 1

        for m in range(k + 1, N):
            for n in range(k):
                # A[m][k] = GEMM (A[k][n], A[m][n], A[m][k])
                dag.add_node(current_node, label="GEMM")

                if (k, n) in last_acted_on:
                    dag.add_edge(last_acted_on[(k, n)], current_node, weight=W)

                if (m, n) in last_acted_on:
                    dag.add_edge(last_acted_on[(m, n)], current_node, weight=W)

                if (m, k) in last_acted_on:
                    dag.add_edge(last_acted_on[(m, k)], current_node, weight=W)

                last_acted_on[(m, k)] = current_node
                current_node += 1

            # A[m][k] = TRSM(A[k][k], A[m][k])
            dag.add_node(current_node, label="TRSM")

            if (k, k) in last_acted_on:
                dag.add_edge(last_acted_on[(k, k)], current_node, weight=W)

            if (m, k) in last_acted_on:
                dag.add_edge(last_acted_on[(m, k)], current_node, weight=W)

            last_acted_on[(m, k)] = current_node
            current_node += 1

    if random_nodes:

        for u, v, data in dag.edges(data=True):
            if u != root:
                data['weight'] = 0

        for u in nx.algorithms.dfs_tree(dag, root):  # Note: we need to use the DSF instead of topo order
            if u == root:
                continue

            # get the input data volume
            input_volume = list(dag.in_edges(u, data=True))[0][2]['weight']

            output_volume = -1
            # if one of the successor nodes has already an input data volume, then use that
            for v in dag.successors(u):
                for x, y, data in dag.in_edges(v, data=True):
                    if data['weight'] != 0:
                        output_volume = data['weight']
                        # print(f"{v} has already an input edge with volume: ", output_volume)
                        break
                if output_volume != -1:
                    break
            else:
                # decide whether this is an elwise/reducer/upsampler node
                ratios = [1 / 4, 1 / 3, 1 / 2, 2, 3, 4]

                coin = random.randint(
                    0, 2)  # generate on average the same number of elwise, downsampler and upsampler nodes
                # output_volume = max(int(np.random.normal(W, W / 2, 1)), 1)
                if coin == 0:
                    # elwise
                    output_volume = input_volume
                else:
                    if max_output_volume != -1:
                        output_volume = max_output_volume + 1
                        while output_volume > max_output_volume:
                            output_volume = max(int(random.choice(ratios) * input_volume), 1)
                    else:
                        output_volume = max(int(random.choice(ratios) * input_volume), 1)
                # print("Out volume =", output_volume)
            # assign this to all output edges
            for _, v, data in dag.out_edges(u, data=True):
                data['weight'] = output_volume

        # Check
        for u in dag.nodes():
            input_volume = -1
            for _, _, data in dag.in_edges(u, data=True):
                if input_volume == -1:
                    input_volume = data['weight']
                else:
                    if input_volume != data['weight']:
                        from utils.visualize import visualize_dag
                        import pdb
                        pdb.set_trace()
                    assert input_volume == data['weight']

            output_volume = -1
            for _, v, data in dag.out_edges(u, data=True):
                if output_volume == -1:
                    output_volume = data['weight']
                else:
                    assert output_volume == data['weight']

    return dag, root, -1  # the sink is not pseudo
