# S-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Utility functions to derive metrics or costs

'''

import networkx as nx
import numpy as np


def build_W_matrix_HEFT(dag):
    """
    Buidls a computation matrix W for HEFT, using a rate-centric DAG.

    Any task takes max{IN-1, OUT-1} + 1 time to compute 

    Attention: this assumes that a node have the same input volume from all its input edge
    TODO: generalize this
    

    :param dag: [description]
    :type dag: [type]
    """


def makespan(tasks_schedule):
    '''
    :return: Returns the schedule length
    '''

    # Assuming that time starts from zero, we have to find the maximum finishing time
    makespan = 0
    for k, job in tasks_schedule.items():
        makespan = max(makespan, job.end_t)
    return makespan


def save_to_csv(filename: str, header: list, data: list):
    '''
        Saves a single list to a csv

    '''
    # TODO
