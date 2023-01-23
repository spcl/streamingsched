# S-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Builds the DAG with two fork-join levels on it.
    The first one has N task, the second M.
'''

import numpy as np
from utils.metrics import *
from utils.graph import *
import random
import math


def build_fork_join_dag(N=4, M=6, W=16, ccr=False):
    '''
    Create an expand collapse graph
    :param N: number of tasks in the first level
    :param M: number of tasks in the second level
    :param W: amount of data sent/received by tasks
    :return: dag, source and sink nodes
    '''

    dag = nx.DiGraph()

    join_node = N + 1
    sink_node = join_node + M + 1

    # All the nodes have weight W

    # pseudoroot
    dag.add_node(0, weight=0)

    # TODO CCR: currently we support elwise/downsampler/splitte/upsamplingr.
    # Therefore the amount of data produced <= aumount of data consumed

    if ccr is False:
        dag.add_node(join_node, weight=W)  #joiner
        dag.add_node(sink_node, weight=W)
        join_node_data = sink_node_data = W
    else:
        #decide at random the amount of data read by the join and sink node
        # The rest of the nodes will be elwise
        ratios = [1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4]
        coin = random.randint(0, 2)
        join_node_data = W if coin == 0 else max(int(random.choice(ratios) * W), 1)
        sink_node_data = join_node_data if coin == 0 else max(int(random.choice(ratios) * join_node_data), 1)
        dag.add_node(join_node, weight=join_node_data)
        dag.add_node(sink_node, weight=sink_node_data)

    #pseudosink
    dag.add_node(sink_node + 1, pseudo=True, weight=0)
    # first level
    for i in range(1, N + 1):
        dag.add_node(i, weight=W)
        dag.add_edge(0, i, weight=W, stream=False)
        dag.add_edge(i, join_node, weight=join_node_data, stream=False)

    #second level
    for i in range(join_node + 1, sink_node):
        dag.add_node(i, weight=join_node_data)
        dag.add_edge(join_node, i, weight=join_node_data, stream=False)
        dag.add_edge(i, sink_node, weight=sink_node_data, stream=False)

    dag.add_edge(sink_node, sink_node + 1, weight=sink_node_data)

    return dag, 0, sink_node + 1
