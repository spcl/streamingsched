# S-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Builds  a DAG having the shape of a Gaussian Elimination
    for an NxN matrix.

    The total number of tasks will be (N^2+N-2)/2
'''

import numpy as np
from utils.metrics import *
from utils.graph import *
import random
import math


def build_gaussian_dag(N=4, W=16, random_nodes=False, minimum_data_volume=0):
    '''
    Builds the dag
    :param N: size of matrix (NxN)
    :param: W: the starting weight
    :param random_nodes: a boolean flag that indicates whether the DAG must have node with randomly generated production rates.
        Otherwise, all tasks are elementwise tasks
    :return: the dag, the root and the sink nodes
    
    '''

    dag = nx.DiGraph()
    root = 0
    dag.add_node(0, weight=0)  # pseudo root
    # No Pseudosink
    curr_node = 1
    ratios = [1 / 4, 1 / 3, 1 / 2, 2, 3, 4]
    # TODO: random_nodes: we currently support elwise/reduce/downsampler/upsampler nodes.

    for i in range(N - 1):
        first_task_of_row = curr_node

        if not random_nodes:
            produced_data = W
        else:
            # for each row, decide if the first task produce the same amount of data or not
            if i == 0:
                produced_data = W
            else:
                coin = random.randint(0, 2)
                if coin == 0:
                    # elwise
                    produced_data = input_data
                else:
                    # downsampler or upsampler
                    produced_data = -1
                    while produced_data < minimum_data_volume:
                        # complete random
                        # produced_data = max(random.randint(int(input_data / 4), int(input_data * 4)), 1)

                        # attempt with rational
                        # produced_data = max(int(round(random.uniform(0.25, 4), 1) * input_data), 1)

                        # normal with mean and std dev
                        # produced_data = max(int(np.random.normal(W, W / 16, 1)), 1)

                        # or wrt starting point (maybe this does not have really sense in practice but is meaningful)
                        # produced_data = max(int(np.random.normal(input_data, input_data / 2, 1)), 1)

                        # fixed ratios
                        produced_data = max(int(random.choice(ratios) * input_data), 1)

        dag.add_node(first_task_of_row, weight=produced_data)

        if i > 0:
            dag.add_edge(first_task_of_row - (N - i), first_task_of_row, weight=produced_data)
        else:
            dag.add_edge(0, first_task_of_row, weight=produced_data)

        curr_node = curr_node + 1

        for j in range(i + 1, N):
            dag.add_node(curr_node, weight=W)
            # add edge from the first task of the row
            dag.add_edge(first_task_of_row, curr_node, weight=produced_data)

            #add edge from the previous row (if any)
            if i > 0:
                dag.add_edge(curr_node - (N - i), curr_node, weight=produced_data)

            curr_node = curr_node + 1
        input_data = produced_data

    return dag, root, curr_node - 1
