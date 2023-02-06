# Streaming-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Scheduling Unit test
'''

from sched.streaming_sched import StreamingScheduler, ScheduleEvent
from sched.utils import print_schedule, build_tasks_schedule_from_pes_schedule
from utils.visualize import visualize_dag, show_schedule_gantt_chart
from sample_graphs import *
import networkx as nx
import numpy as np
import logging
import pytest
from unittest import TestCase


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
    return dag, computation_matrix


def test_checks():
    '''
    Tests checks in initialization
    '''

    # Nodes must have the same input/otput volume for all edges

    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=1)
    dag.add_edge(1, 2, weight=1)
    scheduler = StreamingScheduler(dag, 1)

    # this should raise an assertion error
    dag.add_edge(0, 3, weight=1)
    dag.add_edge(3, 1, weight=2)
    with pytest.raises(AssertionError):
        scheduler = StreamingScheduler(dag, 1)


def test_compute_exec_time_isolation():
    '''
    Tests computation of execution time of task in isolation
    :return:
    '''

    ### 0->1 graph
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=10)
    scheduler = StreamingScheduler(dag, 1)
    result = scheduler._compute_execution_time_isolation(1)
    assert result == 10

    ### 1->3, 2->3, 3->4 graph, 0 pseudo-root
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=5)
    dag.add_edge(0, 2, weight=5)
    dag.add_edge(1, 3, weight=10)
    dag.add_edge(2, 3, weight=10)
    dag.add_edge(3, 4, weight=20)
    scheduler = StreamingScheduler(dag, 1)
    result = scheduler._compute_execution_time_isolation(3)
    assert result == 20

    # Compute execution time of the pseudo-root (which has no input edges)
    result = scheduler._compute_execution_time_isolation(0)
    assert result == 0


def test_ranku():
    '''
    Tests upward rank computation
    '''

    #################################################################
    # Simple graph with 4 nodes, in a romboid shape, and 3 PEs
    #################################################################
    dag = build_dag_4(same_weights=True)
    scheduler = StreamingScheduler(dag, 1)

    scheduler.ranku()
    expected_ranku = [11, 9, 11, 4]  # Pseudo root has execution time == 0

    for rankidx in range(len(expected_ranku)):
        assert dag.nodes()[rankidx]['ranku'] == pytest.approx(expected_ranku[rankidx], 0.001)

    #################################################################
    # More complicated graph with 8 nodes, multi inputs, and 3 PEs
    #################################################################
    dag = build_dag_8(same_weights=True)

    scheduler = StreamingScheduler(dag, 1)

    scheduler.ranku()
    expected_ranku = [26, 25, 13, 26, 8, 14, 14, 3]

    for rankidx in range(len(expected_ranku)):
        assert dag.nodes()[rankidx]['ranku'] == pytest.approx(expected_ranku[rankidx], 0.001)


@pytest.mark.skip(reason="RANKD must be implemented")
def test_rankd():
    '''
    Tests downward rank computation
    '''

    #################################################################
    # Simple graph with 4 nodes, in a romboid shape, and 3 PEs
    #################################################################
    dag, II, Lat = build_dag_4()
    scheduler = StreamingScheduler(dag, II, Lat)

    scheduler.rankd()
    expected_rankd = [0, 8, 18, 21]

    for rankidx in range(len(expected_rankd)):
        assert dag.nodes()[rankidx]['rankd'] == pytest.approx(expected_rankd[rankidx], 0.001)

    #################################################################
    # More complicated graph with 8 nodes, multi inputs, and 3 PEs
    #################################################################
    dag, II, Lat = build_dag_8(3)

    scheduler = StreamingScheduler(dag, II, Lat)

    scheduler.rankd()
    expected_rankd = [0, 3, 2, 11, 6, 16, 21, 25]

    for rankidx in range(len(expected_rankd)):
        assert dag.nodes()[rankidx]['rankd'] == pytest.approx(expected_rankd[rankidx], 0.001)


def test_schedule_no_streaming():
    '''
    Tests Scheduling w/o streaming
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
    pes_schedule, tasks_schedule = scheduler.schedule_dag()

    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=2, pe=0, start_t=0, end_t=8, f_t=2, api=8 / 4),
            ScheduleEvent(task=3, pe=0, start_t=8, end_t=12, f_t=12, api=1),
        ],
        1: [ScheduleEvent(task=1, pe=1, start_t=0, end_t=5, f_t=2,
                          api=5 / 4)]  # note the downsampling ratio is non integer
    }
    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)

    ###################################
    # DAG with 8 tasks and 3 PEs
    ###################################
    dag = build_dag_8(same_weights=True)

    scheduler = StreamingScheduler(dag, num_pes=3)
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()

    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),  # There is always the pseudo-root
            ScheduleEvent(task=3, pe=0, start_t=0, end_t=12, f_t=2, api=12 / 11),
            ScheduleEvent(task=5, pe=0, start_t=12, end_t=23, f_t=16, api=11 / 3),
            ScheduleEvent(task=7, pe=0, start_t=23, end_t=26, f_t=26, api=1),
        ],
        1: [
            ScheduleEvent(task=1, pe=1, start_t=0, end_t=11, f_t=1, api=1),
            ScheduleEvent(task=6, pe=1, start_t=12, end_t=23, f_t=16, api=11 / 3),
        ],
        2: [
            ScheduleEvent(task=2, pe=2, start_t=0, end_t=5, f_t=1, api=1),
            ScheduleEvent(task=4, pe=2, start_t=5, end_t=10, f_t=7, api=5 / 3)
        ]
    }
    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)


def test_schedule_streaming_simple():
    #################################################################
    # Simple graph with 2 elwise nodes (+ pseudo-root), and 2 PEs
    #################################################################
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=5)
    dag.add_edge(1, 2, weight=5, stream=True)
    num_pes = 2

    scheduler = StreamingScheduler(dag, num_pes)
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()

    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=5, f_t=1, api=1),
        ],
        1: [ScheduleEvent(task=2, pe=1, start_t=1, end_t=6, f_t=6, api=1)]
    }

    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)

    ###################################################################
    # Unfeasible scheduling with coscheduling
    #################################################################
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=5)
    dag.add_edge(1, 2, weight=5, stream=True)
    num_pes = 1
    scheduler = StreamingScheduler(dag, num_pes=1)
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()

    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=5, f_t=1, api=1),
            ScheduleEvent(task=2, pe=0, start_t=5, end_t=10, f_t=10, api=1)
        ]
    }

    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)


def test_schedule_streaming_single():
    '''
    Tests for schedule computation with streams.
    Only one edge can be a stream
    Basic topology is the following
        ┌────► 2 ──────────────
      0-│                      ▼
        └────► 1 ────► 3 ────► 4
    '''

    #####################################
    # First case, non streaming
    #####################################

    dag = nx.DiGraph()
    dag.add_node(0, type=0)
    dag.add_node(1, type=0)
    dag.add_node(2, type=0)
    dag.add_node(3, type=0)
    dag.add_node(4, type=0)
    dag.add_edge(0, 1, weight=50, stream=False)
    dag.add_edge(0, 2, weight=50, stream=False)
    dag.add_edge(1, 3, weight=50, stream=False)
    dag.add_edge(3, 4, weight=50, stream=False)
    dag.add_edge(2, 4, weight=50, stream=False)

    num_pes = 2
    num_tasks = len(dag.nodes())

    scheduler = StreamingScheduler(dag, num_pes)
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=50, f_t=1, api=1),
            ScheduleEvent(task=3, pe=0, start_t=50, end_t=100, f_t=51, api=1),
            ScheduleEvent(task=4, pe=0, start_t=100, end_t=150, f_t=150, api=1),
        ],
        1: [ScheduleEvent(task=2, pe=1, start_t=0, end_t=50, f_t=1, api=1)]
    }

    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)

    #####################################
    # Stream between 2 and 4
    #####################################

    dag[2][4]['stream'] = True
    scheduler = StreamingScheduler(dag, num_pes=num_pes)
    pes_schedule, tasks_schedule = scheduler.schedule_dag()

    # Here there is no coscheduling, so 2 can start immediately (same situation as before)
    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=50, f_t=1, api=1),
            ScheduleEvent(task=3, pe=0, start_t=50, end_t=100, f_t=51, api=1),
            ScheduleEvent(task=4, pe=0, start_t=100, end_t=150, f_t=150, api=1),
        ],
        1: [ScheduleEvent(task=2, pe=1, start_t=0, end_t=50, f_t=1, api=1)]
    }

    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)

    #####################################
    # Stream between 1 and 3
    #####################################

    dag[2][4]['stream'] = False
    dag[1][3]['stream'] = True
    scheduler = StreamingScheduler(dag, num_pes=3)
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()

    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=50, f_t=1, api=1),
            ScheduleEvent(task=4, pe=0, start_t=51, end_t=101, f_t=101, api=1),
        ],
        1: [ScheduleEvent(task=2, pe=1, start_t=0, end_t=50, f_t=1, api=1)],
        2: [ScheduleEvent(task=3, pe=2, start_t=1, end_t=51, f_t=2, api=1)]
    }

    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)

    ################################################
    # Different topology: expand/collapse.
    ################################################

    dag = nx.DiGraph()
    N = 2
    M = 4
    W = 16
    join_node = N + 1
    sink_node = join_node + M + 1
    # pseudoroot
    dag.add_node(0)
    dag.add_node(join_node)  # joiner
    dag.add_node(sink_node)
    # first level
    for i in range(1, N + 1):
        dag.add_edge(0, i, weight=W, stream=False)
        dag.add_edge(i, join_node, weight=W, stream=False)

    # second level
    for i in range(join_node + 1, sink_node):
        dag.add_edge(join_node, i, weight=W, stream=False)
        dag.add_edge(i, sink_node, weight=W, stream=False)

    num_pes = 3

    # add stream
    dag[2][3]['stream'] = True
    dag[1][3]['stream'] = True

    scheduler = StreamingScheduler(dag, num_pes=num_pes)
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()

    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=16, f_t=1, api=1),
            ScheduleEvent(task=4, pe=0, start_t=17, end_t=33, f_t=18, api=1),
            ScheduleEvent(task=7, pe=0, start_t=33, end_t=49, f_t=34, api=1),
            ScheduleEvent(task=8, pe=0, start_t=49, end_t=65, f_t=65, api=1),
        ],
        1: [
            ScheduleEvent(task=2, pe=1, start_t=0, end_t=16, f_t=1, api=1),
            ScheduleEvent(task=5, pe=1, start_t=17, end_t=33, f_t=18, api=1),
        ],
        2: [
            ScheduleEvent(task=3, pe=2, start_t=1, end_t=17, f_t=2, api=1),
            ScheduleEvent(task=6, pe=2, start_t=17, end_t=33, f_t=18, api=1),
        ]
    }

    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)


def test_schedule_streaming_chain():
    '''
    Chain of N tasks, all of them are Elwise-like tasks and work with an M-elements input.
    '''
    N = 4
    M = 8
    dag = nx.DiGraph()
    # pseudoroot
    dag.add_edge(0, 1, weight=M)
    for i in range(N - 1):
        dag.add_edge(i + 1, i + 2, weight=M, stream=True)

    num_pes = N

    scheduler = StreamingScheduler(dag, num_pes=num_pes)
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()

    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=8, f_t=1, api=1),
        ],
        1: [ScheduleEvent(task=2, pe=1, start_t=1, end_t=9, f_t=2, api=1)],
        2: [ScheduleEvent(task=3, pe=2, start_t=2, end_t=10, f_t=3, api=1)],
        3: [ScheduleEvent(task=4, pe=3, start_t=3, end_t=11, f_t=11, api=1)],
    }

    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)

    # Change weight to have a downsampler and check that production rate satisfy this.
    # The following node will be then an upsampler

    dag[1][2]['weight'] = 4
    scheduler = StreamingScheduler(dag, num_pes=num_pes)
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=8, f_t=2, api=2),
        ],
        1: [ScheduleEvent(task=2, pe=1, start_t=2, end_t=10, f_t=3, api=1)],
        2: [ScheduleEvent(task=3, pe=2, start_t=3, end_t=11, f_t=4, api=1)],
        3: [ScheduleEvent(task=4, pe=3, start_t=4, end_t=12, f_t=12, api=1)],
    }

    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)

    #Let's have another upsampler and another node
    dag[3][4]['weight'] = 16
    dag.add_edge(4, 5, weight=4, stream=True)
    scheduler = StreamingScheduler(dag, num_pes=num_pes)

    scheduler.streaming_interval_analysis()
    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=13, f_t=2, api=13 / 4),
            ScheduleEvent(task=5, pe=0, start_t=13, end_t=26, f_t=26, api=13 / 4),
        ],
        1: [ScheduleEvent(task=2, pe=1, start_t=2, end_t=17, f_t=3, api=15 / 8)],
        2: [ScheduleEvent(task=3, pe=2, start_t=3, end_t=19, f_t=4, api=1)],
        3: [ScheduleEvent(task=4, pe=3, start_t=4, end_t=20, f_t=8, api=4)],
    }
    pes_schedule, tasks_schedule = scheduler.schedule_dag()

    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)

    #########
    # Ad-hoc chain: sequence of downsamplers
    # NOTE that we can not really detect that some node can not actually stream
    # but we run it anyway for error check

    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=128, stream=False)
    dag.add_edge(1, 2, weight=128, stream=True)
    dag.add_edge(2, 3, weight=32, stream=True)
    dag.add_edge(3, 4, weight=32, stream=True)
    dag.add_edge(4, 5, weight=10, stream=True)
    dag.add_edge(5, 6, weight=10, stream=True)
    dag.add_edge(6, 7, weight=10, stream=False)

    scheduler = StreamingScheduler(dag, num_pes=2)
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()

    # TODO

    # print_schedule(pes_schedule, "PE")
    # expected_pes_schedule = {
    #     0: [
    #         ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
    #         ScheduleEvent(task=1, pe=0, start_t=0, end_t=128, f_t=1, api=1),
    #         ScheduleEvent(task=3, pe=0, start_t=128, end_t=21, f_t=21, api=(3 * 4 + 1) / 4),
    #         ScheduleEvent(task=5, pe=0, start_t=8, end_t=21, f_t=21, api=(3 * 4 + 1) / 4),
    #         ScheduleEvent(task=7, pe=0, start_t=8, end_t=21, f_t=21, api=(3 * 4 + 1) / 4),
    #     ],
    #     1: [
    #         ScheduleEvent(task=2, pe=1, start_t=1, end_t=129, f_t=5, api=4),
    #         ScheduleEvent(task=4, pe=1, start_t=2, end_t=10, f_t=3, api=1),
    #         ScheduleEvent(task=6, pe=1, start_t=2, end_t=10, f_t=3, api=1)
    #     ],
    # }
    # TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    # TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)


def test_schedule_streaming_multi():
    '''
    Tests on DAGs with more than one streaming connection.
    All tasks are elwise

    Basic topology is the following

               ┌─────► 5 ─────
               │             ▼
        ┌────► 2 ────► 4───► 6
      0-│                    ^
        └────► 1 ────► 3 ────│
    :return:
    '''

    #####################################
    # First case, non streaming
    #####################################

    dag = nx.DiGraph()
    dag.add_node(0, type=0)  # Pseudo-root
    dag.add_node(1, type=0)
    dag.add_node(2, type=0)
    dag.add_node(3, type=0)
    dag.add_node(4, type=0)
    dag.add_node(5, type=0)
    dag.add_node(6, type=0)
    dag.add_edge(0, 1, weight=50, stream=False)
    dag.add_edge(0, 2, weight=50, stream=False)
    dag.add_edge(1, 3, weight=50, stream=False)
    dag.add_edge(3, 6, weight=50, stream=False)
    dag.add_edge(4, 6, weight=50, stream=False)
    dag.add_edge(5, 6, weight=50, stream=False)
    dag.add_edge(2, 5, weight=50, stream=False)
    dag.add_edge(2, 4, weight=50, stream=False)

    num_pes = 2
    scheduler = StreamingScheduler(dag, num_pes=2)
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()

    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=50, f_t=1, api=1),
            ScheduleEvent(task=3, pe=0, start_t=50, end_t=100, f_t=51, api=1),
            ScheduleEvent(task=5, pe=0, start_t=100, end_t=150, f_t=101, api=1),
            ScheduleEvent(task=6, pe=0, start_t=150, end_t=200, f_t=200, api=1),
        ],
        1: [
            ScheduleEvent(task=2, pe=1, start_t=0, end_t=50, f_t=1, api=1),
            ScheduleEvent(task=4, pe=1, start_t=50, end_t=100, f_t=51, api=1)
        ]
    }
    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)

    ##############################################################
    # Let's add 2 streaming communications that can run in parallel
    ##############################################################
    dag[1][3]['stream'] = True
    dag[2][4]['stream'] = True
    scheduler = StreamingScheduler(dag, num_pes=num_pes)
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    # show_schedule_gantt_chart(pes_schedule)

    # TODO: this is not actually tested but insist on this to improve the heuristic:
    # - we may need to trim
    # - to remove empty spaces


def test_streaming_depth():
    '''
    Small test on streaming depth computation
    '''

    #### All elwises

    # single chain
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=16)
    dag.add_edge(1, 2, weight=16)
    dag.add_edge(2, 3, weight=16)
    dag.add_edge(3, 4, weight=16)

    scheduler = StreamingScheduler(dag, num_pes=1)
    assert scheduler.get_streaming_depth() == 20

    # two paths, one longer than the other
    dag.add_edge(0, 5, weight=16)
    dag.add_edge(5, 6, weight=16)
    dag.add_edge(6, 7, weight=16)
    dag.add_edge(7, 8, weight=16)
    dag.add_edge(8, 9, weight=16)

    dag.add_node(10, pseudo=True)
    dag.add_edge(9, 10, weight=16)
    dag.add_edge(4, 10, weight=16)

    scheduler = StreamingScheduler(dag, num_pes=1)
    assert scheduler.get_streaming_depth() == 21

    #### All downsamplers

    # two paths, one longer than the other, one with a larger starting volume
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=64)
    dag.add_edge(1, 2, weight=16)
    dag.add_edge(2, 3, weight=4)
    dag.add_edge(3, 4, weight=2)
    dag.add_edge(0, 5, weight=32)
    dag.add_edge(5, 6, weight=8)
    dag.add_edge(6, 7, weight=8)
    dag.add_edge(7, 8, weight=8)
    dag.add_edge(8, 9, weight=8)

    dag.add_node(10, pseudo=True)
    dag.add_edge(9, 10, weight=1)
    dag.add_edge(4, 10, weight=1)

    scheduler = StreamingScheduler(dag, num_pes=1)
    assert scheduler.get_streaming_depth() == 68

    #### Mixed
    print("---------")
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=64)
    dag.add_edge(1, 2, weight=16, stream=True)
    dag.add_edge(2, 3, weight=4, stream=True)
    dag.add_edge(3, 4, weight=2, stream=True)
    dag.add_edge(0, 5, weight=32, stream=True)
    dag.add_edge(5, 6, weight=32, stream=True)
    dag.add_edge(6, 7, weight=128, stream=True)
    dag.add_edge(7, 8, weight=8, stream=True)
    dag.add_edge(8, 9, weight=8, stream=True)

    dag.add_node(10, pseudo=True)
    dag.add_edge(9, 10, weight=1, stream=True)
    dag.add_edge(4, 10, weight=1, stream=True)

    scheduler = StreamingScheduler(dag, num_pes=10)
    scheduler.streaming_interval_analysis()
    streaming_blocks = scheduler.get_streaming_blocks()
    pes_schedule, tasks_schedule = scheduler.gang_schedule(streaming_blocks)
    print_schedule(pes_schedule, "PE")

    print(scheduler.get_streaming_depth())


def test_buffer_nodes():
    """
        Tests with buffer nodes
    """

    ######### Chain

    ## chain reduction
    dag = nx.DiGraph()
    dag.add_node(0)
    dag.add_edge(0, 1, weight=128, stream=False)
    dag.add_edge(1, 2, weight=64, stream=True)
    dag.add_edge(2, 3, weight=32, stream=False)  # Buffer node must have non-stream output edge
    dag.add_edge(3, 4, weight=16, stream=True)

    #Let's assume that node 2 is buffer node
    scheduler = StreamingScheduler(dag, num_pes=2, buffer_nodes={2})
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()

    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=128, f_t=2, api=2),
            ScheduleEvent(task=2, pe=0, start_t=128, end_t=128, f_t=128, api=0),  # the buffer
            ScheduleEvent(task=3, pe=0, start_t=128, end_t=160, f_t=130, api=2),
        ],
        1: [ScheduleEvent(task=4, pe=1, start_t=130, end_t=161, f_t=161, api=31 / 16)]
    }
    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)

    ######## Multi path
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=128, stream=False)
    dag.add_edge(1, 2, weight=32, stream=True)
    dag.add_edge(2, 3, weight=4, stream=True)
    dag.add_edge(2, 6, weight=4, stream=True)
    dag.add_edge(3, 4, weight=1, stream=False)  # buffer node
    dag.add_edge(4, 5, weight=2, stream=True)
    dag.add_edge(5, 7, weight=4, stream=True)
    dag.add_edge(6, 7, weight=4, stream=False)  # buffer node
    dag.add_edge(7, 8, weight=4, stream=True)
    # Node 3 and 6 are buffer nodes
    scheduler = StreamingScheduler(dag, num_pes=3, buffer_nodes={3, 6})
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=128, f_t=4, api=4.0),
            ScheduleEvent(task=3, pe=0, start_t=129, end_t=129, f_t=129, api=0.0),  # buffer
            ScheduleEvent(task=6, pe=0, start_t=129, end_t=129, f_t=129, api=0.0),  # buffer
            ScheduleEvent(task=4, pe=0, start_t=129, end_t=132, f_t=130, api=1.5),
            ScheduleEvent(task=8, pe=0, start_t=132, end_t=136, f_t=136, api=1.0)
        ],
        1: [
            ScheduleEvent(task=2, pe=1, start_t=4, end_t=129, f_t=33, api=(129 - 4) / 4),
            ScheduleEvent(task=5, pe=1, start_t=130, end_t=134, f_t=131, api=1.0)
        ],
        2: [ScheduleEvent(task=7, pe=2, start_t=131, end_t=135, f_t=132, api=1.0)]
    }
    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)
