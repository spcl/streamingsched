# S-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Builds the DAG for FFT.
    Parametric on N, which must be a power of 2
'''

from utils.metrics import *
from utils.graph import *
import math
import random


def build_fft_dag(N=8, W=128, random_nodes=None, only_butterfly=False):
    '''
    Creates a (random) Canonical FFT Task Graph
    :param N: number of input of the FFT (must be a power of 2)
    :param W: starting weight
    :param random_nodes: a boolean flag that indicates whether the DAG must have node with randomly generated production rates.
        Otherwise, all tasks are elementwise tasks sending/receiving W elements
    :param only_butterfly: generates only the butterfly component
    :return:
    '''

    # Check that N is power of 2
    assert (N & (N - 1) == 0) and N != 0

    dag = nx.DiGraph()

    dag.add_node(0, weight=0)  #pseudo root

    pseudo_sink_node = N * (int(math.log2(N))) + 2 * N - 1
    dag.add_node(pseudo_sink_node, pseudo=True, weight=0)
    node_counter = 1
    ratios = [1 / 4, 1 / 3, 1 / 2, 2, 3, 4]

    if not only_butterfly:
        ### First part, recursive calls:
        # There are 2xN -1 recursive calls
        recursive_tasks = 2
        # print("How many levels: ", int(math.log2(N)))
        for t in range(int(math.log2(N))):
            # for each level, decide the amount of data that is produced
            if random_nodes is None:
                produced_data = W
            else:
                if t == 0:
                    produced_data = W
                else:
                    coin = random.randint(0, 2)
                    if coin == 0:
                        # elwise
                        produced_data = produced_data
                    else:
                        # downsampler/upsampler
                        produced_data = max(int(random.choice(ratios) * produced_data), 1)
            for i in range(recursive_tasks):
                dag.add_node(node_counter, weight=produced_data)
                curr_node = node_counter
                # add edge
                if t == 0:
                    # add edge from pseudo root
                    dag.add_edge(0, curr_node, weight=produced_data)
                else:
                    # add edges from the previous level
                    prev = int((curr_node - 1) / 2)

                    # same node at the previous level
                    dag.add_edge(prev, curr_node, weight=produced_data)

                node_counter = node_counter + 1
            recursive_tasks *= 2

    ### Second part, butterfly:

    start_range = 0 if only_butterfly else 1
    for t in range(start_range, int(math.log2(N)) + 1):

        # for each level, decide the amount of data that is produced
        if random_nodes is None:
            produced_data = W
        else:
            if t == 0:
                produced_data = W
            else:
                coin = random.randint(0, 2)
                if coin == 0:
                    # elwise
                    produced_data = produced_data
                else:
                    # downsampler/upsampler
                    produced_data = max(int(random.choice(ratios) * produced_data), 1)
        for i in range(N):
            dag.add_node(node_counter, weight=produced_data)
            curr_node = node_counter
            # add edge
            if t == 0 and only_butterfly:
                # add edge from pseudo root
                dag.add_edge(0, curr_node, weight=produced_data)
            else:
                # add edges from the previous level
                prev = curr_node - N

                # same node at the previous level
                dag.add_edge(prev, curr_node, weight=produced_data)
                pos = i % (2**t)
                if pos < 2**(t - 1):
                    prev_level_node = prev + 2**(t - 1)
                else:
                    prev_level_node = prev - 2**(t - 1)
                dag.add_edge(prev_level_node, curr_node, weight=produced_data)

            #if last level, add edge to pseudoroot
            if t == int(math.log2(N)):
                dag.add_edge(curr_node, pseudo_sink_node, weight=produced_data)
            node_counter = node_counter + 1

    return dag, 0, pseudo_sink_node
