# S-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
Simple tests for homoegenous list based scheduling (currently they are the same of HEFT)
"""

from sched.eft import schedule_dag, ranku, EFTScheduleEvent
import networkx as nx
import numpy as np
from unittest import TestCase
from sched.streaming_sched import StreamingScheduler, ScheduleEvent
from sched.utils import print_schedule, build_tasks_schedule_from_pes_schedule
from sched.utils import build_W_matrix_HEFT
from utils.metrics import makespan
from sample_graphs import *


def are_schedule_equivalent(stream_tasks_schedule, eft_tasks_schedule):
    # Do something more precise? Like starting time?
    return makespan(stream_tasks_schedule) == makespan(eft_tasks_schedule)


def test_equivalence():
    '''
        This test verifies that EFT and SSched returns the same schedule
        if there are no streaming edges
    '''

    #################################################################
    # Simple graph with 4 nodes, in a romboid shape, and 2 PEs
    #################################################################
    # TODO: API may be subject to change

    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=5)
    dag.add_edge(0, 2, weight=8)
    dag.add_edge(1, 3, weight=4)
    dag.add_edge(2, 3, weight=4)
    num_pes = 2
    num_tasks = len(dag.nodes())

    scheduler = StreamingScheduler(dag, num_pes=num_pes)
    scheduler.streaming_interval_analysis()
    stream_pes_schedule, stream_tasks_schedule = scheduler.schedule_dag()

    # build the EFT scheduling
    W = build_W_matrix_HEFT(dag, 0, 4, {}, 1)
    eft_pes_schedule, eft_tasks_schedule = schedule_dag(dag, W, num_pes)

    # check that the schedules are equivalent (either each task has the same running time, and the makespan is the same)
    assert are_schedule_equivalent(stream_tasks_schedule, eft_tasks_schedule)

    dag = build_dag_8(same_weights=True)

    scheduler = StreamingScheduler(dag, num_pes=3)
    scheduler.streaming_interval_analysis()
    stream_pes_schedule, stream_tasks_schedule = scheduler.schedule_dag()

    W = build_W_matrix_HEFT(dag, 0, 8, {}, 1)
    eft_pes_schedule, eft_tasks_schedule = schedule_dag(dag, W, 3)

    # check that the schedules are equivalent (either each task has the same running time, and the makespan is the same)
    assert are_schedule_equivalent(stream_tasks_schedule, eft_tasks_schedule)

    ### Chain with buffer node

    dag = nx.DiGraph()
    dag.add_node(0)
    dag.add_edge(0, 1, weight=128, stream=False)
    dag.add_edge(1, 2, weight=64, stream=False)
    dag.add_edge(2, 3, weight=32, stream=False)  # Buffer node must have non-stream output edge
    dag.add_edge(3, 4, weight=16, stream=False)
    buffer_nodes = {2}
    #Let's assume that node 2 is buffer node
    scheduler = StreamingScheduler(dag, num_pes=2, buffer_nodes=buffer_nodes)
    scheduler.streaming_interval_analysis()
    stream_pes_schedule, stream_tasks_schedule = scheduler.schedule_dag()
    W = build_W_matrix_HEFT(dag, 0, 5, buffer_nodes, 1)
    eft_pes_schedule, eft_tasks_schedule = schedule_dag(dag, W, 2)

    # check that the schedules are equivalent (either each task has the same running time, and the makespan is the same)
    assert are_schedule_equivalent(stream_tasks_schedule, eft_tasks_schedule)
