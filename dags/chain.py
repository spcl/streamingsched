# Streaming-Sched: Streaming Scheduling for DataFlow Architectures
# Copyright (c) 2023 ETH-Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Methods to build a DAG (and corresponding costs) composed by single chain of tasks
'''
import networkx as nx
import numpy as np
from utils.metrics import *
from utils.graph import *
import random
import math


def build_chain_dag(N=16, W=128, random_nodes=True, minim_data_volume=0, maximum_output_volume=-1):
    '''
    Create a canonical task graph with a chain of N tasks
    :param N: number of nodes
    :param W: starting weight
    :param random_nodes: a boolean flag that indicates whether the DAG must have node with randomly generated production rates.
        Otherwise, all tasks are elementwise tasks sending/receiving W elements
    :param min data volume: minim data volume to have in the edge
    :param maximum_output_volume: maximum output volume of a node
    :return: the dag, the root and the pseudo sink nodes
    '''

    dag = nx.DiGraph()
    # pseudo nodes
    pseudo_root = 0
    pseudo_sink = N + 1
    dag.add_node(pseudo_root, weight=0)
    dag.add_node(1, weight=W)

    dag.add_edge(0, 1, weight=W)
    input_data = W
    for i in range(N - 1):

        if random_nodes:
            # We support downsampling/elwise/upsampling nodes with fixed ratios.
            ratios = [1 / 4, 1 / 3, 1 / 2, 2, 3, 4]

            #decide whether we keep the same amount of data or not
            coin = random.randint(0, 2)
            if coin == 0:
                output_volume = input_data
            else:
                #downsampler/upsampling
                if maximum_output_volume == -1:
                    output_volume = -1
                    while output_volume < minim_data_volume:
                        output_volume = max(int(random.choice(ratios) * input_data), 1)
                else:
                    output_volume = maximum_output_volume + 1
                    while output_volume > maximum_output_volume:
                        output_volume = max(int(random.choice(ratios) * input_data), 1)
        else:
            output_volume = input_data

        dag.add_node(i + 2, weight=W)
        dag.add_edge(i + 1, i + 2, weight=output_volume, stream=False)
        input_data = output_volume

    # Pseudo_sink
    dag.add_node(pseudo_sink, pseudo=True, weight=0)
    dag.add_edge(N, pseudo_sink, weight=output_volume)

    return dag, pseudo_root, pseudo_sink
