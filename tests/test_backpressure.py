# S-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Tests for backpressure analysis. This version uses streaming_intervals
'''

from sched.streaming_sched import StreamingScheduler, ScheduleEvent
from sched.utils import print_schedule, build_tasks_schedule_from_pes_schedule
from utils.visualize import visualize_dag, show_schedule_gantt_chart
import networkx as nx
import numpy as np
import logging
import pytest
from unittest import TestCase


def test_backpressure_chain_streaming_intervals():
    '''
    We have a streaming chain of task, in different combinations

    '''

    ### All elwise
    N = 4
    M = 16
    dag = nx.DiGraph()

    # pseudoroot
    dag.add_edge(0, 1, weight=M)
    for i in range(N - 1):
        dag.add_edge(i + 1, i + 2, weight=M, stream=True)

    scheduler = StreamingScheduler(dag, num_pes=N)
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()

    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=16, f_t=1, api=1),
        ],
        1: [
            ScheduleEvent(task=2, pe=1, start_t=1, end_t=17, f_t=2, api=1),
        ],
        2: [
            ScheduleEvent(task=3, pe=2, start_t=2, end_t=18, f_t=3, api=1),
        ],
        3: [
            ScheduleEvent(task=4, pe=3, start_t=3, end_t=19, f_t=19, api=1),
        ],
    }
    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)

    # In this case we have a downsampler with ratio 8 followed by an upsampler with ratio 2
    # The streaming intervals are 1,8,2,1
    # - task 1 is a downsampler: it outputs every 8 unit of times (starts at t=0, first result at 8)
    # - task 2, is an upsampler with ratio 4: it outputs the first element at time 9. Outputs follow a streaming
    #       interval of 2 (because the next node is another upsampler).
    # - task 3 is an upsampler with ratio 2: it outpus the first element at time 10. It has to output 16 elements
    #       it finishes at time 25
    # - task 4 is an elwise

    dag[1][2]['weight'] = 2
    dag[2][3]['weight'] = 8
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    # print_schedule(pes_schedule, "PE")

    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=16, f_t=8, api=8),
        ],
        1: [
            ScheduleEvent(task=2, pe=1, start_t=8, end_t=23, f_t=9, api=15 / 8),
        ],
        2: [
            ScheduleEvent(task=3, pe=2, start_t=9, end_t=25, f_t=10, api=1),
        ],
        3: [
            ScheduleEvent(task=4, pe=3, start_t=10, end_t=26, f_t=26, api=1),
        ],
    }
    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)

    # Let's have an upsampler (NOTE that 0 is the pseudo-root)
    # 0 -- (16) --> 1 -- (16) --> 2 -- (32) --> 3 -- (16) --> 4
    # Streaming intervals: 1, 2, 1, 2
    dag[1][2]['weight'] = 16
    dag[2][3]['weight'] = 32

    scheduler.streaming_interval_analysis()
    # for u, v, data in dag.edges(data=True):
    #     print(f"[{u}, {v}]: {data['streaming_interval']}")

    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=31, f_t=1, api=31 / 16),
        ],
        1: [
            ScheduleEvent(task=2, pe=1, start_t=1, end_t=33, f_t=2, api=1),
        ],
        2: [
            ScheduleEvent(task=3, pe=2, start_t=2, end_t=34, f_t=4, api=2),
        ],
        3: [
            ScheduleEvent(task=4, pe=3, start_t=4, end_t=35, f_t=35, api=31 / 16),
        ],
    }
    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)

    # Same scenario as before, but this time the edge 1 ->2 is not streaming:
    # In this case, 1 will be not backpressured, but 2 can start only when 1 finishes

    # In this case it may use one PE less
    dag[1][2]['stream'] = False
    scheduler.streaming_interval_analysis()
    # for u, v, data in dag.edges(data=True):
    #     print(f"[{u}, {v}]: {data['streaming_interval']}")

    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    # print_schedule(pes_schedule, "PE")

    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=16, f_t=1, api=1),
            ScheduleEvent(task=2, pe=0, start_t=16, end_t=48, f_t=17, api=1),
        ],
        1: [
            ScheduleEvent(task=3, pe=1, start_t=17, end_t=49, f_t=19, api=2),
        ],
        2: [
            ScheduleEvent(task=4, pe=2, start_t=19, end_t=50, f_t=50, api=31 / 16),
        ],
        3: [],
    }
    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)

    # # Ad-hoc chain:
    # # TODO this does not work if we don't use streaming interval (despite it does not have upsamplers. The time of sink is incorrect)
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=128, stream=False)
    dag.add_edge(1, 2, weight=32, stream=True)
    dag.add_edge(2, 3, weight=32, stream=True)
    dag.add_edge(3, 4, weight=8, stream=True)
    dag.add_edge(4, 5, weight=8, stream=True)

    scheduler = StreamingScheduler(dag, num_pes=5)
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()

    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=128, f_t=4, api=4),
        ],
        1: [
            ScheduleEvent(task=2, pe=1, start_t=4, end_t=129, f_t=5, api=125 / 32),
        ],
        2: [
            ScheduleEvent(task=3, pe=2, start_t=5, end_t=130, f_t=18, api=125 / 8),  # it has to wait 4 inputs to output
        ],
        3: [ScheduleEvent(task=4, pe=3, start_t=18, end_t=131, f_t=19, api=113 / 8)],
        4: [ScheduleEvent(task=5, pe=4, start_t=19, end_t=132, f_t=132, api=113 / 8)]
    }
    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)

    # Chain with upsampler at the end
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=16, stream=False)
    dag.add_edge(1, 2, weight=16, stream=True)
    dag.add_edge(2, 3, weight=2, stream=True)
    dag.add_edge(3, 4, weight=2, stream=True)
    dag.add_edge(4, 5, weight=32, stream=True)

    scheduler = StreamingScheduler(dag, num_pes=5)
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()

    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=31, f_t=1,
                          api=31 / 16),  # streaming interval 2 (TODO: should't f_t be =2?)
        ],
        1: [
            ScheduleEvent(task=2, pe=1, start_t=1, end_t=32, f_t=16, api=31 / 2),
        ],
        2: [
            ScheduleEvent(task=3, pe=2, start_t=16, end_t=33, f_t=17, api=17 / 2),
        ],
        3: [ScheduleEvent(task=4, pe=3, start_t=17, end_t=49, f_t=18, api=1)],
        4: [ScheduleEvent(task=5, pe=4, start_t=18, end_t=50, f_t=50, api=1)]
    }
    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)

    # # TODO tests with consum/prod ratio that are not integers (see streaming interval)

    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=128, stream=False)
    dag.add_edge(1, 2, weight=32, stream=True)
    dag.add_edge(2, 3, weight=8, stream=True)
    dag.add_edge(3, 4, weight=2, stream=True)
    dag.add_edge(4, 5, weight=1, stream=True)
    scheduler = StreamingScheduler(dag, num_pes=5)
    scheduler.streaming_interval_analysis()
    for u, v, data in dag.edges(data=True):
        print(f"[{u}, {v}]: {data['streaming_interval']}")
    pes_schedule, tasks_schedule = scheduler.schedule_dag()

    print_schedule(pes_schedule, "PE")


def test_additional_chain():

    # Special cases

    ### All elwise
    N = 4
    M = 16
    dag = nx.DiGraph()

    # pseudoroot
    dag.add_edge(0, 1, weight=M)
    for i in range(N - 1):
        dag.add_edge(i + 1, i + 2, weight=M, stream=True)

    dag[1][2]['weight'] = 2
    dag[2][3]['weight'] = 8
    dag[2][3]['stream'] = False

    scheduler = StreamingScheduler(dag, num_pes=N)

    scheduler.streaming_interval_analysis()

    pes_schedule, tasks_schedule = scheduler.schedule_dag()

    # This particular case is not manually covered in compute_average_exec_time_in_schedule: node 2 is an upsampler with ratio 4, it should finish at time 20
    # Only two PEs will be used because of the non-streaming edge

    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=16, f_t=8, api=8),
            ScheduleEvent(task=3, pe=0, start_t=20, end_t=36, f_t=21, api=1),
        ],
        1: [
            ScheduleEvent(task=2, pe=1, start_t=8, end_t=20, f_t=9, api=12 / 8),
            ScheduleEvent(task=4, pe=1, start_t=21, end_t=37, f_t=37, api=1),
        ],
        2: [],
        3: [],
    }

    # print_schedule(pes_schedule, "PE")

    ################
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=128)
    dag.add_edge(1, 2, weight=512)
    dag.add_edge(2, 3, weight=128, stream=True)
    dag.add_edge(3, 4, weight=256)
    # dag.add_node(5, pseudo=True)
    dag.add_edge(4, 5, weight=256, stream=True)

    scheduler = StreamingScheduler(dag, num_pes=2)

    scheduler.streaming_interval_analysis()
    for i, (u, v, data) in enumerate(dag.edges(data=True)):
        print(f"{u} -> {v} : {data['streaming_interval']}")

    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    print_schedule(pes_schedule, "PE")


def test_backpressure_multi_input_streaming_intervals():
    '''
    Basic topology is the following
           ┌────► 3 ──────────────
      0─►1-│                      ▼
           └────► 2 ────► 4 ────► 5
    '''

    # TODO We should deal with the fact that 5 starts only when 4 produces the first data, and therefore 3 must wait for this...
    # Then fix the tested schedules
    dag = nx.DiGraph()
    dag.add_node(0)
    dag.add_node(1)
    dag.add_node(2)
    dag.add_node(3)
    dag.add_node(4)
    dag.add_node(5)
    dag.add_edge(0, 1, weight=8)  #pseudo root
    dag.add_edge(1, 2, weight=8, stream=True)
    dag.add_edge(1, 3, weight=8, stream=True)
    dag.add_edge(2, 4, weight=8, stream=True)
    dag.add_edge(4, 5, weight=8, stream=True)
    dag.add_edge(3, 5, weight=8, stream=True)

    num_pes = 5
    scheduler = StreamingScheduler(dag, num_pes=num_pes)
    scheduler.streaming_interval_analysis()
    # for u, v, data in dag.edges(data=True):
    #     print(f"[{u}, {v}]: {data['streaming_interval']}")

    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    # print_schedule(pes_schedule, "PE")

    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=8, f_t=1, api=1),
        ],
        1: [
            ScheduleEvent(task=2, pe=1, start_t=1, end_t=9, f_t=2, api=1),
        ],
        2: [
            ScheduleEvent(task=3, pe=2, start_t=1, end_t=9, f_t=2,
                          api=1),  # This one does not take into account that 5 starts later
        ],
        3: [
            ScheduleEvent(task=4, pe=3, start_t=2, end_t=10, f_t=3, api=1),
        ],
        4: [
            ScheduleEvent(task=5, pe=4, start_t=3, end_t=11, f_t=11, api=1),
        ],
    }
    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)

    # Now, node 2 is an upsampler (and node 4) a downsampler. So it will slow-down node 1 and also node 3, 5
    dag[2][4]['weight'] = 16

    scheduler.streaming_interval_analysis()

    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    # print_schedule(pes_schedule, "PE")

    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=15, f_t=1, api=15 / 8),
        ],
        1: [
            ScheduleEvent(task=2, pe=1, start_t=1, end_t=17, f_t=2, api=1),
        ],
        2: [
            ScheduleEvent(task=4, pe=2, start_t=2, end_t=18, f_t=4, api=2),  # this is a downsampler
        ],
        3: [
            ScheduleEvent(task=3, pe=3, start_t=1, end_t=16, f_t=2,
                          api=15 / 8),  # This one does not take into account that 5 starts later
        ],
        4: [
            ScheduleEvent(task=5, pe=4, start_t=4, end_t=19, f_t=19, api=15 / 8),
        ],
    }
    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)
