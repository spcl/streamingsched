# Streaming-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Tests for gang scheduling analysis. 
'''

from sched.streaming_sched import StreamingScheduler, ScheduleEvent
from sched.utils import print_schedule, build_tasks_schedule_from_pes_schedule
from utils.visualize import visualize_dag, show_schedule_gantt_chart
import networkx as nx
import numpy as np
import logging
import pytest
from unittest import TestCase


def set_streams_from_streaming_paths(dag, streaming_paths):

    # set all edges to non streaming
    for src, dst, data in dag.edges(data=True):
        data['stream'] = False

    for t in streaming_paths:
        for i in range(len(t) - 1):
            if dag.has_edge(t[i], t[i + 1]):
                dag[t[i]][t[i + 1]]['stream'] = True
            else:
                print(f"Something is going wrong: edge ({t[i]}, {t[i+1]}) does not exists.")


def test_chain():
    '''
    Linear chain of tasks, with different configuration.
    Streaming blocks and path are manually defined
    :return:
    '''

    ###################################
    ### All elwise, no backpressure
    ###################################

    N = 4
    M = 16
    dag = nx.DiGraph()

    # pseudoroot
    dag.add_edge(0, 1, weight=M)
    for i in range(N - 1):
        dag.add_edge(i + 1, i + 2, weight=M, stream=True)

    scheduler = StreamingScheduler(dag, num_pes=N)

    streaming_blocks = [[0, 1, 2, 3, 4]]
    streaming_paths = [[0, 1], [1, 2], [2, 3], [3, 4]]
    set_streams_from_streaming_paths(dag, streaming_paths)

    # Gang scheduling relies on streaming intervals computed considering streaming blocks
    pes_schedule, tasks_schedule = scheduler.gang_schedule(streaming_blocks)

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

    # Less PEs

    scheduler = StreamingScheduler(dag, num_pes=2)

    streaming_blocks = [[0, 1, 2], [3, 4]]
    streaming_paths = [[0, 1], [1, 2], [3, 4]]
    set_streams_from_streaming_paths(dag, streaming_paths)

    # Gang scheduling relies on streaming intervals computed considering streaming blocks

    pes_schedule, tasks_schedule = scheduler.gang_schedule(streaming_blocks)

    # In this case, the second streaming block can only start when the first finished
    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=16, f_t=1, api=1),
            ScheduleEvent(task=3, pe=0, start_t=17, end_t=33, f_t=18, api=1),
        ],
        1: [
            ScheduleEvent(task=2, pe=1, start_t=1, end_t=17, f_t=2, api=1),
            ScheduleEvent(task=4, pe=1, start_t=18, end_t=34, f_t=34, api=1),
        ]
    }

    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)

    ################################
    # Different nodes but no backpressure
    ################################
    # In this case we have a downsampler with ratio 8 followed by an upsampler with ratio 2
    # The streaming intervals are 1,8,2,1
    # - task 1 is a downsampler: it outputs every 8 unit of times (starts at t=0, first result at 8)
    # - task 2, is an upsampler with ratio 4: it outputs the first element at time 9. Outputs follow a streaming
    #       interval of 2 (because the next node is another upsampler).
    # - task 3 is an upsampler with ratio 2: it outpust the first element at time 10. It has to output 16 elements
    #       it finishes at time 25
    # - task 4 is an elwise

    dag[1][2]['weight'] = 2
    dag[2][3]['weight'] = 8
    scheduler = StreamingScheduler(dag, num_pes=N)

    streaming_blocks = [[0, 1, 2, 3, 4]]
    streaming_paths = [[0, 1], [1, 2], [2, 3], [3, 4]]
    set_streams_from_streaming_paths(dag, streaming_paths)

    pes_schedule, tasks_schedule = scheduler.gang_schedule(streaming_blocks)
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

    # Use less PEs

    scheduler = StreamingScheduler(dag, num_pes=2)

    streaming_blocks = [[0, 1, 2], [3, 4]]
    streaming_paths = [[0, 1], [1, 2], [3, 4]]
    set_streams_from_streaming_paths(dag, streaming_paths)

    # Gang scheduling relies on streaming intervals computed considering streaming blocks
    pes_schedule, tasks_schedule = scheduler.gang_schedule(streaming_blocks)

    # for u, v, data in dag.edges(data=True):
    #     print(f"{u} -> {v} : {data['streaming_interval']}")

    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=16, f_t=8, api=8),
            ScheduleEvent(task=3, pe=0, start_t=20, end_t=36, f_t=21, api=1),
        ],
        1: [
            ScheduleEvent(task=2, pe=1, start_t=8, end_t=20, f_t=9, api=12 / 8),
            ScheduleEvent(task=4, pe=1, start_t=21, end_t=37, f_t=37, api=1),
        ]
    }

    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)

    # Let's have an upsampler (NOTE that 0 is the pseudo-root)
    # 0 -- (16) --> 1 -- (16) --> 2 -- (32) --> 3 -- (16) --> 4
    # Streaming intervals: 1, 2, 1, 2

    dag[1][2]['weight'] = 16
    dag[2][3]['weight'] = 32

    scheduler = StreamingScheduler(dag, num_pes=2)
    streaming_blocks = [[0, 1, 2], [3, 4]]
    streaming_paths = [[0, 1], [1, 2], [3, 4]]
    set_streams_from_streaming_paths(dag, streaming_paths)

    # Gang scheduling relies on streaming intervals computed considering streaming blocks
    # scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.gang_schedule(streaming_blocks)

    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=31, f_t=1, api=31 / 16),
            ScheduleEvent(task=3, pe=0, start_t=33, end_t=65, f_t=35, api=2),
        ],
        1: [
            ScheduleEvent(task=2, pe=1, start_t=1, end_t=33, f_t=2, api=1),
            ScheduleEvent(task=4, pe=1, start_t=35, end_t=66, f_t=66, api=31 / 16),
        ],
    }

    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)

    # ####################################
    # # Ad-hoc chain:
    # ####################################

    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=128, stream=False)
    dag.add_edge(1, 2, weight=32, stream=True)
    dag.add_edge(2, 3, weight=32, stream=True)
    dag.add_edge(3, 4, weight=8, stream=True)
    dag.add_edge(4, 5, weight=8, stream=True)

    # Let's assume 3 PEs
    scheduler = StreamingScheduler(dag, num_pes=3)
    streaming_blocks = [[0, 1, 2, 3], [4, 5]]
    streaming_paths = [[0, 1], [1, 2], [2, 3], [4, 5]]
    set_streams_from_streaming_paths(dag, streaming_paths)

    # Gang scheduling relies on streaming intervals computed considering streaming blocks
    # scheduler.streaming_interval_analysis()
    # Streaming intervals are 4,4,1,1
    pes_schedule, tasks_schedule = scheduler.gang_schedule(streaming_blocks)

    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=128, f_t=4, api=4),
            ScheduleEvent(task=4, pe=0, start_t=130, end_t=138, f_t=131, api=1)
        ],
        1: [
            ScheduleEvent(task=2, pe=1, start_t=4, end_t=129, f_t=5, api=125 / 32),
            ScheduleEvent(task=5, pe=1, start_t=131, end_t=139, f_t=139, api=1)
        ],
        2: [
            ScheduleEvent(task=3, pe=2, start_t=5, end_t=130, f_t=18, api=125 / 8),
        ]
    }

    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)

    # ###############################
    # # Chain with upsampler at the end, number of PE sufficient to have all streaming
    # #############################################
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=16, stream=False)
    dag.add_edge(1, 2, weight=16, stream=True)
    dag.add_edge(2, 3, weight=2, stream=True)
    dag.add_edge(3, 4, weight=2, stream=True)
    dag.add_edge(4, 5, weight=32, stream=True)

    scheduler = StreamingScheduler(dag, num_pes=5)
    streaming_blocks = [[0, 1, 2, 3, 4, 5]]
    streaming_paths = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
    set_streams_from_streaming_paths(dag, streaming_paths)

    # Gang scheduling relies on streaming intervals computed considering streaming blocks
    pes_schedule, tasks_schedule = scheduler.gang_schedule(streaming_blocks)

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


def test_multi_input():
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

    scheduler = StreamingScheduler(dag, num_pes=3)
    streaming_blocks = [[0, 1, 2, 3], [4, 5]]
    streaming_paths = [[0, 1], [1, 2], [1, 3], [4, 5]]
    set_streams_from_streaming_paths(dag, streaming_paths)

    # Gang scheduling relies on streaming intervals computed considering streaming blocks
    pes_schedule, tasks_schedule = scheduler.gang_schedule(streaming_blocks)
    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=8, f_t=1, api=1),
            ScheduleEvent(task=4, pe=0, start_t=9, end_t=17, f_t=10, api=1)
        ],
        1: [
            ScheduleEvent(task=3, pe=1, start_t=1, end_t=9, f_t=2, api=1),
            ScheduleEvent(task=5, pe=1, start_t=10, end_t=18, f_t=18, api=1),
        ],
        2: [
            ScheduleEvent(task=2, pe=2, start_t=1, end_t=9, f_t=2, api=1),
        ]
    }
    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)

    ######################################################
    # In this case, edge between 4 and 5 is not streaming
    # Note: this should not be the case in a streaming block but we want to support it anyway
    #######################################################
    streaming_blocks = [[0, 1, 2, 3], [4, 5]]
    streaming_paths = [[0, 1], [1, 2], [1, 3]]
    set_streams_from_streaming_paths(dag, streaming_paths)

    # Gang scheduling relies on streaming intervals computed considering streaming blocks
    pes_schedule, tasks_schedule = scheduler.gang_schedule(streaming_blocks, analyze=True)

    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=8, f_t=1, api=1),
            ScheduleEvent(task=4, pe=0, start_t=9, end_t=17, f_t=10, api=1)
        ],
        1: [
            ScheduleEvent(task=3, pe=1, start_t=1, end_t=9, f_t=2, api=1),
            ScheduleEvent(task=5, pe=1, start_t=17, end_t=25, f_t=25, api=1),
        ],
        2: [
            ScheduleEvent(task=2, pe=2, start_t=1, end_t=9, f_t=2, api=1),
        ]
    }

    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)


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
    dag.add_edge(2, 3, weight=32, stream=True)
    dag.add_edge(3, 4, weight=16, stream=False)  # Buffer node must have non-stream output edge
    dag.add_edge(4, 5, weight=16, stream=True)

    #Let's assume that node 2 is buffer node
    scheduler = StreamingScheduler(dag, num_pes=2, buffer_nodes={3})
    pes_schedule, tasks_schedule = scheduler.gang_schedule(scheduler.get_streaming_blocks())

    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=128, f_t=2, api=2),
            ScheduleEvent(task=4, pe=0, start_t=129, end_t=145, f_t=130, api=1.0),
        ],
        1: [
            ScheduleEvent(task=2, pe=1, start_t=2, end_t=129, f_t=5, api=(129 - 2) / 32),
            ScheduleEvent(task=3, pe=1, start_t=129, end_t=129, f_t=129, api=0.0),
            ScheduleEvent(task=5, pe=1, start_t=130, end_t=146, f_t=146, api=1.0)
        ]
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
    scheduler = StreamingScheduler(dag, num_pes=4, buffer_nodes={3, 6})
    pes_schedule, tasks_schedule = scheduler.gang_schedule(scheduler.get_streaming_blocks())

    print_schedule(pes_schedule, "PE")
    expected_pes_schedule = {
        0: [
            ScheduleEvent(task=0, pe=0, start_t=0, end_t=0, f_t=0, api=0),
            ScheduleEvent(task=1, pe=0, start_t=0, end_t=128, f_t=4, api=4.0),
            ScheduleEvent(task=4, pe=0, start_t=129, end_t=132, f_t=130, api=1.5)
        ],
        1: [
            ScheduleEvent(task=2, pe=1, start_t=4, end_t=129, f_t=33, api=31.25),
            ScheduleEvent(task=6, pe=1, start_t=129, end_t=129, f_t=129,
                          api=0.0),  #buffer allocate on the same PE of latest predecessors
            ScheduleEvent(task=3, pe=1, start_t=129, end_t=129, f_t=129, api=0.0),  #buffer
            ScheduleEvent(task=5, pe=1, start_t=130, end_t=134, f_t=131, api=1.0),
        ],
        2: [ScheduleEvent(task=7, pe=2, start_t=131, end_t=135, f_t=132, api=1.0)],
        3: [ScheduleEvent(task=8, pe=3, start_t=132, end_t=136, f_t=136, api=1.0)]
    }
    TestCase().assertDictEqual(expected_pes_schedule, pes_schedule)
    TestCase().assertDictEqual(build_tasks_schedule_from_pes_schedule(pes_schedule), tasks_schedule)
