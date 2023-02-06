# Streaming-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Creates a DAG for a Matrix-Matrix Multiplication
    Given A, a NxK matrix, B, a KxM matrix, and C, and NxM matrix,
    we want to compute

            C += A @ B

    By following the naive approach, we organize the computation in a cube of
    NxMxK tasks, connected in such a way to share partial results (Cij) and input data (either A, or B).
    Each task perform a multiply and add.
'''

import numpy as np
from utils.metrics import *
from utils.graph import *
import random
import math


def build_cholesky_mmm(N=2, M=2, K=2, W=16):
    '''
    Builds a Cholesky task graph
    :param N: number of tiles
    :return: the dag, the root and the sink nodes
    '''

    dag = nx.DiGraph()
    root = 0

    # Keep a dictionary for easy look up of tasks
    tasks_dictionary = dict()  # (n,m,k) to task

    curr_node = 1
    for n in range(N):
        for m in range(M):
            for k in range(K):

                dag.add_node(curr_node, weight=W, label=f"A{n}{k}_B{k}{m}")

                # Dependencies:
                # TODO: understand if the first two are needed. They push the data without re-reading it
                # this task depends from the A value pushed by the predecessor along the M axis
                if m > 0:
                    dag.add_edge(tasks_dictionary[(n, m - 1, k)], curr_node, weight=W)

                # it depends from the B value pushed along the N axis
                if n > 0:
                    dag.add_edge(tasks_dictionary[(n - 1, m, k)], curr_node, weight=W)

                # it depends from the C value pushed along the k axis
                if k > 0:
                    dag.add_edge(tasks_dictionary[(n, m, k - 1)], curr_node, weight=W)

                tasks_dictionary[(n, m, k)] = curr_node
                curr_node += 1

    dag.add_node(root, weight=0)  # pseudo root
    # add edges from the root and sink
    for n in get_source_nodes(dag):
        if n != root:
            dag.add_edge(root, n, weight=W)

    sink_node = curr_node
    dag.add_node(sink_node, pseudo=True, weight=0)
    for n in get_sink_nodes(dag):
        if n != sink_node:
            dag.add_edge(n, sink_node, weight=W)

    return dag, root, sink_node
