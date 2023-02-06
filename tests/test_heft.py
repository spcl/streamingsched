# Streaming-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    HEFT Scheduling Tests
"""

from sched.heft import schedule_dag, ranku, HEFTScheduleEvent
import networkx as nx
import numpy as np
from unittest import TestCase
from sched.streaming_sched import StreamingScheduler, ScheduleEvent
from sched.utils import print_schedule, build_tasks_schedule_from_pes_schedule
from sched.utils import build_W_matrix_HEFT
from utils.metrics import makespan
from sample_graphs import *


def are_schedule_equivalent(stream_tasks_schedule, heft_tasks_schedule):
    # Do something more precise? Like starting time?
    return makespan(stream_tasks_schedule) == makespan(heft_tasks_schedule)


def paper_matrices():
    # Create task graph: use topology in HEFT original paper
    dag_adj_matrix = np.array([[0, 18, 12, 9, 11, 14, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 19, 16, 0],
                               [0, 0, 0, 0, 0, 0, 23, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 27, 23, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 13, 0], [0, 0, 0, 0, 0, 0, 0, 15, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 17], [0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 13], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    dag = nx.from_numpy_matrix(dag_adj_matrix, create_using=nx.DiGraph)
    # Computation matrix: 10 (tasks) x 3 (PEs)
    computation_matrix = np.array([[14, 16, 9], [13, 19, 18], [11, 13, 19], [13, 8, 17], [12, 13, 10], [13, 16, 9],
                                   [7, 15, 11], [5, 11, 14], [18, 12, 20], [21, 7, 16]])
    communication_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    communication_startup = np.zeros(3)
    return dag, computation_matrix, communication_matrix, communication_startup


def test_ranku():
    '''
    Tests upward rank computation
    '''

    # Since there is no comm cost, here what is important is the crical path
    # computed considering average execution times
    dag, computation_matrix, communication_matrix, communication_startup = paper_matrices()
    ranku(dag, computation_matrix, communication_matrix, communication_startup)

    # With original comm cost
    expected_ranku = [108, 77, 80, 80, 69, 63.333, 42.667, 35.667, 44.333, 14.667]

    for rankidx in range(len(expected_ranku)):
        TestCase().assertAlmostEqual(dag.nodes()[rankidx]['ranku'], expected_ranku[rankidx], delta=0.001)

    # Without comm cost
    dag, computation_matrix, communication_matrix, communication_startup = paper_matrices()
    expected_ranku = [61, 48, 40, 44, 43, 37.3333, 25.6666, 24.6666, 31.3333, 14.6666]

    ranku(dag, computation_matrix)
    for rankidx in range(len(expected_ranku)):
        TestCase().assertAlmostEqual(dag.nodes()[rankidx]['ranku'], expected_ranku[rankidx], delta=0.001)


def test_schedule():
    '''
    Tests schedule, without and with considering commcost
    :return:
    '''
    expected_task_schedules = {
        0: HEFTScheduleEvent(task=0, pe=2, start_t=0, end_t=9),
        1: HEFTScheduleEvent(task=1, pe=0, start_t=9, end_t=22),
        2: HEFTScheduleEvent(task=2, pe=1, start_t=17, end_t=30),
        3: HEFTScheduleEvent(task=3, pe=1, start_t=9, end_t=17),
        4: HEFTScheduleEvent(task=4, pe=2, start_t=9, end_t=19),
        5: HEFTScheduleEvent(task=5, pe=2, start_t=19, end_t=28),
        6: HEFTScheduleEvent(task=6, pe=2, start_t=30, end_t=41),
        7: HEFTScheduleEvent(task=7, pe=1, start_t=30, end_t=41),
        8: HEFTScheduleEvent(task=8, pe=0, start_t=22, end_t=40),
        9: HEFTScheduleEvent(task=9, pe=1, start_t=41, end_t=48)
    }

    expected_pes_schedules = {
        0:
        [HEFTScheduleEvent(task=1, pe=0, start_t=9, end_t=22),
         HEFTScheduleEvent(task=8, pe=0, start_t=22, end_t=40)],
        1: [
            HEFTScheduleEvent(task=3, pe=1, start_t=9, end_t=17),
            HEFTScheduleEvent(task=2, pe=1, start_t=17, end_t=30),
            HEFTScheduleEvent(task=7, pe=1, start_t=30, end_t=41),
            HEFTScheduleEvent(task=9, pe=1, start_t=41, end_t=48)
        ],
        2: [
            HEFTScheduleEvent(task=0, pe=2, start_t=0, end_t=9),
            HEFTScheduleEvent(task=4, pe=2, start_t=9, end_t=19),
            HEFTScheduleEvent(task=5, pe=2, start_t=19, end_t=28),
            HEFTScheduleEvent(task=6, pe=2, start_t=30, end_t=41),
        ]
    }

    dag, computation_matrix, communication_matrix, communication_startup = paper_matrices()

    # Scheduling without communication costs
    pes_schedules, task_schedules = schedule_dag(dag, computation_matrix)
    TestCase().assertDictEqual(expected_task_schedules, task_schedules)
    TestCase().assertDictEqual(expected_pes_schedules, pes_schedules)

    # Scheduling with communication costs

    expected_pes_schedules = {
        0: [
            HEFTScheduleEvent(task=1, pe=0, start_t=27.0, end_t=40.0),
            HEFTScheduleEvent(task=7, pe=0, start_t=57.0, end_t=62.0)
        ],
        1: [
            HEFTScheduleEvent(task=3, pe=1, start_t=18.0, end_t=26.0),
            HEFTScheduleEvent(task=5, pe=1, start_t=26.0, end_t=42.0),
            HEFTScheduleEvent(task=8, pe=1, start_t=56.0, end_t=68.0),
            HEFTScheduleEvent(task=9, pe=1, start_t=73.0, end_t=80.0)
        ],
        2: [
            HEFTScheduleEvent(task=0, pe=2, start_t=0, end_t=9.0),
            HEFTScheduleEvent(task=2, pe=2, start_t=9.0, end_t=28.0),
            HEFTScheduleEvent(task=4, pe=2, start_t=28.0, end_t=38.0),
            HEFTScheduleEvent(task=6, pe=2, start_t=38.0, end_t=49.0)
        ]
    }
    expected_task_sched = {
        0: HEFTScheduleEvent(task=0, pe=2, start_t=0, end_t=9.0),
        1: HEFTScheduleEvent(task=1, pe=0, start_t=27.0, end_t=40.0),
        2: HEFTScheduleEvent(task=2, pe=2, start_t=9.0, end_t=28.0),
        3: HEFTScheduleEvent(task=3, pe=1, start_t=18.0, end_t=26.0),
        4: HEFTScheduleEvent(task=4, pe=2, start_t=28.0, end_t=38.0),
        5: HEFTScheduleEvent(task=5, pe=1, start_t=26.0, end_t=42.0),
        6: HEFTScheduleEvent(task=6, pe=2, start_t=38.0, end_t=49.0),
        7: HEFTScheduleEvent(task=7, pe=0, start_t=57.0, end_t=62.0),
        8: HEFTScheduleEvent(task=8, pe=1, start_t=56.0, end_t=68.0),
        9: HEFTScheduleEvent(task=9, pe=1, start_t=73.0, end_t=80.0)
    }

    # Scheduling without communication costs
    dag, computation_matrix, communication_matrix, communication_startup = paper_matrices()
    pes_schedules, task_schedules = schedule_dag(dag, computation_matrix, communication_matrix, communication_startup)

    TestCase().assertDictEqual(expected_task_sched, task_schedules)
    TestCase().assertDictEqual(expected_pes_schedules, pes_schedules)


def test_equivalence():
    '''
        This test verifies that HEFT and SSched returns the same schedule
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

    # build the HEFT scheduling
    W = build_W_matrix_HEFT(dag, 0, 4, {}, 2)
    heft_pes_schedule, heft_tasks_schedule = schedule_dag(dag, W)

    # check that the schedules are equivalent (either each task has the same running time, and the makespan is the same)
    assert are_schedule_equivalent(stream_tasks_schedule, heft_tasks_schedule)

    dag = build_dag_8(same_weights=True)

    scheduler = StreamingScheduler(dag, num_pes=3)
    scheduler.streaming_interval_analysis()
    stream_pes_schedule, stream_tasks_schedule = scheduler.schedule_dag()

    W = build_W_matrix_HEFT(dag, 0, 8, {}, 3)
    heft_pes_schedule, heft_tasks_schedule = schedule_dag(dag, W)

    # check that the schedules are equivalent (either each task has the same running time, and the makespan is the same)
    assert are_schedule_equivalent(stream_tasks_schedule, heft_tasks_schedule)

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
    W = build_W_matrix_HEFT(dag, 0, 5, buffer_nodes, 2)
    heft_pes_schedule, heft_tasks_schedule = schedule_dag(dag, W)

    # check that the schedules are equivalent (either each task has the same running time, and the makespan is the same)
    assert are_schedule_equivalent(stream_tasks_schedule, heft_tasks_schedule)
