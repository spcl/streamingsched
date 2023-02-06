# Streaming-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Test on streaming interval computation
'''

from sched.streaming_sched import StreamingScheduler, ScheduleEvent
from sched.utils import print_schedule, build_tasks_schedule_from_pes_schedule
from utils.visualize import visualize_dag, show_schedule_gantt_chart
import networkx as nx
import numpy as np
import logging
import pytest
from unittest import TestCase
from fractions import Fraction


def test_chain_streaming_interval_analysis():
    '''
    Chain of N tasks
    '''

    ### All elwise
    N = 2
    M = 16
    dag = nx.DiGraph()
    # pseudoroot
    dag.add_edge(0, 1, weight=M, stream=True)
    for i in range(N - 1):
        dag.add_edge(i + 1, i + 2, weight=M, stream=True)

    num_pes = N

    scheduler = StreamingScheduler(dag, num_pes=num_pes)

    # In this case we have a downsampler with ratio 8 followed by an upsampler with ratio 2
    # The streaming interval of the output edge from upsampler is 2
    dag[1][2]['weight'] = 32
    scheduler.streaming_interval_analysis()
    expected_streaming_intervals = [2, 1]
    for i, (u, v, data) in enumerate(dag.edges(data=True)):
        print(u, v, data)
        assert data['streaming_interval'] == expected_streaming_intervals[i]

    ### All elwise
    N = 4
    M = 16
    dag = nx.DiGraph()
    # pseudoroot
    dag.add_edge(0, 1, weight=M)  # NOTE: the first edge is not streaming
    for i in range(N - 1):
        dag.add_edge(i + 1, i + 2, weight=M, stream=True)

    num_pes = N

    scheduler = StreamingScheduler(dag, num_pes=num_pes)

    # In this case we have a downsampler with ratio 8 followed by an upsampler with ratio 2
    # The streaming interval of the output edge from upsampler is 2
    dag[1][2]['weight'] = 2
    dag[2][3]['weight'] = 8

    scheduler.streaming_interval_analysis()
    expected_streaming_intervals = [1, 8, 2, 1]
    for i, (u, v, data) in enumerate(dag.edges(data=True)):
        print(u, v, data)
        assert data['streaming_interval'] == expected_streaming_intervals[i]

    dag[1][2]['weight'] = 32
    scheduler.streaming_interval_analysis()

    expected_streaming_intervals = [1, 1, 4, 2]
    for i, (u, v, data) in enumerate(dag.edges(data=True)):
        print(u, v, data)
        assert data['streaming_interval'] == expected_streaming_intervals[i]

    # Let's have more upsampler one after the other
    dag[2][3]['weight'] = 64
    scheduler.streaming_interval_analysis()

    expected_streaming_intervals = [1, 2, 1, 4]
    for i, (u, v, data) in enumerate(dag.edges(data=True)):
        assert data['streaming_interval'] == expected_streaming_intervals[i]

    # Another chain: only downsampler, the first node reads from memory but it still produces
    # data every 4 ticks
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=16, stream=False)
    dag.add_edge(1, 2, weight=4, stream=True)
    dag.add_edge(2, 3, weight=1, stream=True)
    dag.add_edge(3, 4, weight=1, stream=True)

    scheduler = StreamingScheduler(dag, num_pes=5)
    scheduler.streaming_interval_analysis()

    expected_streaming_intervals = [1, 4, 16, 16]
    # for u, v, data in dag.edges(data=True):
    # print(f"[{u}, {v}]: {data['streaming_interval']}")

    for i, (u, v, data) in enumerate(dag.edges(data=True)):
        print(u, v, data)
        assert data['streaming_interval'] == expected_streaming_intervals[i]

    # Chain with data volume not a multiple of each other

    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=16, stream=True)
    dag.add_edge(1, 2, weight=16, stream=True)
    dag.add_edge(2, 3, weight=8, stream=True)
    dag.add_edge(3, 4, weight=7, stream=True)
    dag.add_edge(4, 5, weight=23, stream=True)

    scheduler = StreamingScheduler(dag, num_pes=5)
    scheduler.streaming_interval_analysis()

    expected_streaming_intervals = [23 / 16, 23 / 16, 23 / 8, 23 / 7, 1]
    for u, v, data in dag.edges(data=True):
        print(f"[{u}, {v}]: {data['streaming_interval']}")

    for i, (u, v, data) in enumerate(dag.edges(data=True)):
        TestCase().assertAlmostEqual(data['streaming_interval'], expected_streaming_intervals[i], delta=0.001)


def test_multi_input_streaming_interval_analysis():
    '''
    Basic topology is the following
        ┌────► 2 ──────────────
      0-│                      ▼
        └────► 1 ────► 3 ────► 4
    '''

    dag = nx.DiGraph()
    dag.add_node(0, type=0)
    dag.add_node(1, type=0)
    dag.add_node(2, type=0)
    dag.add_node(3, type=0)
    dag.add_node(4, type=0)
    dag.add_edge(0, 1, weight=8, stream=True)
    dag.add_edge(0, 2, weight=8, stream=True)
    dag.add_edge(1, 3, weight=8, stream=True)
    dag.add_edge(3, 4, weight=8, stream=True)
    dag.add_edge(2, 4, weight=8, stream=True)

    scheduler = StreamingScheduler(dag, num_pes=1)
    scheduler.streaming_interval_analysis()

    for u, v, data in dag.edges(data=True):
        assert data['streaming_interval'] == 1

    # Node 1 in a downsamlpler

    dag[1][3]['weight'] = 4

    scheduler.streaming_interval_analysis()

    # # Edges order [(0,1), (0,2), (1,3), (3,4), (2,4)]
    expected_streaming_intervals = [1, 1, 2, 1, 1]
    for i, (u, v, data) in enumerate(dag.edges(data=True)):
        assert data['streaming_interval'] == expected_streaming_intervals[i]

    # Node 2 and 3 are upsamplers
    dag[2][4]['weight'] = 16
    dag[3][4]['weight'] = 16
    # visualize_dag(dag)
    scheduler.streaming_interval_analysis()

    expected_streaming_intervals = [2, 2, 4, 1, 1]
    for i, (u, v, data) in enumerate(dag.edges(data=True)):
        assert data['streaming_interval'] == expected_streaming_intervals[i]

    ################################################
    # Different topology: expand/collapse, where bottom
    # nodes are upsamplers
    ################################################

    dag = nx.DiGraph()
    N = 2
    M = 3
    W = 8
    join_node = N + 1
    sink_node = join_node + M + 1
    # pseudoroot
    dag.add_node(0)
    dag.add_node(join_node)  # joiner
    dag.add_node(sink_node)
    # first level
    for i in range(1, N + 1):
        dag.add_edge(0, i, weight=W, stream=True)
        dag.add_edge(i, join_node, weight=W, stream=True)

    # second level
    for i in range(join_node + 1, sink_node):
        dag.add_edge(join_node, i, weight=W, stream=True)
        dag.add_edge(i, sink_node, weight=4 * W, stream=True)

    scheduler = StreamingScheduler(dag, num_pes=1)
    scheduler.streaming_interval_analysis()
    expected_streaming_intervals = [4, 4, 4, 4, 4, 4, 4, 1, 1, 1]
    for i, (u, v, data) in enumerate(dag.edges(data=True)):
        assert data['streaming_interval'] == expected_streaming_intervals[i]
    pass


def test_asymmetric_paths_streaming_interval_analysis():
    '''
    We want to test DAGs with asymmetric paths
    '''
    from dags import gaussian_elimination

    dag, source_, exit_node = gaussian_elimination.build_gaussian_dag(4, W=16)
    scheduler = StreamingScheduler(dag, num_pes=1)
    scheduler.streaming_interval_analysis()

    # set all edges to streaming
    for u, v, data in dag.edges(data=True):
        data['stream'] = True

    for u, v, data in dag.edges(data=True):
        assert data['streaming_interval'] == 1

    # Let's add downsamplers (nodes 3,4, and 5 are downsampler, the rest elwise)
    # Edges from downsamplers are non streaming, so they do not affect downstream nodes

    dag[3][6]['weight'] = 8
    dag[4][7]['weight'] = 8
    dag[5][6]['weight'] = 8
    dag[5][7]['weight'] = 8
    dag[6][8]['weight'] = 8
    dag[7][9]['weight'] = 8
    dag[8][9]['weight'] = 8
    dag[3][6]['stream'] = False
    dag[4][7]['stream'] = False
    dag[5][6]['stream'] = False
    dag[5][7]['stream'] = False
    scheduler.streaming_interval_analysis()
    expected_streaming_intervals = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    for i, (u, v, data) in enumerate(dag.edges(data=True)):
        assert data['streaming_interval'] == expected_streaming_intervals[i]

    # Let's have upsampler now: nodes 4, and 7 are upsampler with factor 2x, node 6 with factor 4
    # We also remove the edge between 5 and 7 to have a more various dag
    # Edges 6->8, 8->9 and 7->9 are non streaming, yet they should backpressure
    dag, source_, exit_node = gaussian_elimination.build_gaussian_dag(4, W=16)
    # set all edges to streaming
    for u, v, data in dag.edges(data=True):
        data['stream'] = True
    dag.remove_edge(5, 7)
    dag[4][7]['weight'] = 32
    dag[6][8]['weight'] = 64
    dag[7][9]['weight'] = 64
    dag[8][9]['weight'] = 64
    dag[6][8]['stream'] = False
    dag[8][9]['stream'] = False
    dag[7][9]['stream'] = False

    scheduler = StreamingScheduler(dag, num_pes=1)
    scheduler.streaming_interval_analysis()
    for u, v, data in dag.edges(data=True):
        print(f"[{u}, {v}]: {data['streaming_interval']}")
    expected_streaming_intervals = [4, 4, 4, 4, 4, 4, 2, 4, 1, 1, 1]
    for i, (u, v, data) in enumerate(dag.edges(data=True)):
        assert data['streaming_interval'] == expected_streaming_intervals[i]

    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=128, stream=False)
    dag.add_edge(0, 1, weight=128, stream=False)
    dag.add_edge(1, 4, weight=32, stream=True)
    dag.add_edge(0, 2, weight=32, stream=False)
    dag.add_edge(2, 3, weight=32, stream=True)
    dag.add_edge(2, 4, weight=32, stream=True)
    dag.add_edge(4, 6, weight=96, stream=True)
    dag.add_edge(3, 5, weight=96, stream=True)
    dag.add_edge(5, 6, weight=96, stream=True)
    scheduler = StreamingScheduler(dag, num_pes=1)
    scheduler.streaming_interval_analysis()
    # visualize_dag(dag)
    for u, v, data in dag.edges(data=True):
        print(f"[{u}, {v}]: {data['streaming_interval']}")
    expected_streaming_intervals = [1, 1, 4, Fraction(4, 3), 4, 4, Fraction(4, 3), Fraction(4, 3)]
    for i, (u, v, data) in enumerate(dag.edges(data=True)):
        assert data['streaming_interval'] == expected_streaming_intervals[i]

    dag = nx.DiGraph()
    dag.add_node(0)
    dag.add_node(7, pseudo=True)
    dag.add_edge(0, 5, weight=6, stream=False)
    dag.add_edge(5, 6, weight=6, stream=True)
    dag.add_edge(0, 1, weight=2, stream=False)
    dag.add_edge(1, 2, weight=2, stream=True)
    dag.add_edge(2, 3, weight=4, stream=True)
    dag.add_edge(3, 4, weight=12, stream=True)
    dag.add_edge(4, 6, weight=6, stream=True)
    dag.add_edge(6, 7, weight=6, stream=False)
    scheduler = StreamingScheduler(dag, num_pes=1)
    scheduler.streaming_interval_analysis()
    for u, v, data in dag.edges(data=True):
        print(f"[{u}, {v}]: {data['streaming_interval']}")
    expected_streaming_intervals = [1, 1, 2, 1, 6, 3, 1, 2]
    for i, (u, v, data) in enumerate(dag.edges(data=True)):
        assert data['streaming_interval'] == expected_streaming_intervals[i]

    dag = nx.DiGraph()
    dag.add_node(0)
    dag.add_edge(0, 4, weight=1536, stream=False)
    dag.add_edge(0, 2, weight=1536, stream=False)
    dag.add_edge(4, 5, weight=48, stream=True)
    dag.add_edge(0, 2, weight=1536, stream=False)
    dag.add_edge(2, 3, weight=96, stream=True)
    dag.add_edge(0, 1, weight=384, stream=False)
    dag.add_edge(1, 3, weight=96, stream=True)
    dag.add_edge(3, 5, weight=48, stream=True)
    dag.add_edge(5, 6, weight=16, stream=True)
    scheduler = StreamingScheduler(dag, num_pes=1)
    scheduler.streaming_interval_analysis()
    expected_streaming_intervals = [1, 1, 1, 32, 16, 96, 32, 16]
    for i, (u, v, data) in enumerate(dag.edges(data=True)):
        assert data['streaming_interval'] == expected_streaming_intervals[i]

    # From Cholesky
    dag = nx.DiGraph()
    dag.add_node(0)
    dag.add_node(9, pseudo=True)
    dag.add_edge(0, 5, weight=32, stream=False)
    dag.add_edge(0, 7, weight=32, stream=False)
    dag.add_edge(7, 8, weight=24, stream=True)
    dag.add_edge(0, 3, weight=32, stream=False)
    dag.add_edge(5, 6, weight=24, stream=True)
    dag.add_edge(3, 4, weight=24, stream=True)
    dag.add_edge(0, 2, weight=96, stream=False)
    dag.add_edge(0, 1, weight=96, stream=False)
    dag.add_edge(2, 6, weight=24, stream=True)
    dag.add_edge(2, 8, weight=24, stream=True)
    dag.add_edge(8, 9, weight=3, stream=False)
    dag.add_edge(1, 4, weight=24, stream=True)
    dag.add_edge(1, 6, weight=24, stream=True)
    dag.add_edge(6, 9, weight=12, stream=False)
    dag.add_edge(4, 9, weight=6, stream=False)
    scheduler = StreamingScheduler(dag, num_pes=1)
    scheduler.streaming_interval_analysis()
    expected_streaming_intervals = [1, 1, 1, 1, 1, 4, 4, 1, 4, 1, 1, 4, 4, 4, 4]
    for i, (u, v, data) in enumerate(dag.edges(data=True)):
        assert data['streaming_interval'] == expected_streaming_intervals[i]


def test_buffer_node():
    '''
        Tests with DAGs having 1+ buffer node
    '''

    ######### Chain

    ## chain reduction
    dag = nx.DiGraph()
    dag.add_node(0)
    dag.add_edge(0, 1, weight=128, stream=False)
    dag.add_edge(1, 2, weight=64, stream=True)
    dag.add_edge(2, 3, weight=32, stream=True)
    dag.add_edge(3, 4, weight=16, stream=True)

    #Let's assume that node 2 is buffer node
    scheduler = StreamingScheduler(dag, num_pes=1, buffer_nodes={2})
    scheduler.streaming_interval_analysis()

    expected_streaming_intervals = [1, 2, 1, 2]
    for i, (u, v, data) in enumerate(dag.edges(data=True)):
        assert data['streaming_interval'] == expected_streaming_intervals[i]

    # Let'assume that 2 produces more data than it reads. Being a buffer node, will not increase the
    # streaming interval of previous nodes
    dag[2][3]['weight'] = 1024
    scheduler.streaming_interval_analysis()
    expected_streaming_intervals = [1, 2, 1, 64]

    for i, (u, v, data) in enumerate(dag.edges(data=True)):
        assert data['streaming_interval'] == expected_streaming_intervals[i]

    # More than one buffer nodes
    dag.add_edge(4, 5, weight=16, stream=True)
    dag.add_edge(5, 6, weight=4, stream=True)
    dag.add_edge(6, 7, weight=8, stream=True)
    scheduler = StreamingScheduler(dag, num_pes=1, buffer_nodes={2, 5})
    scheduler.streaming_interval_analysis()

    # Note: since we are not setting the streaming interval for source/buff nodes (5->6), this will be 1 as well
    expected_streaming_intervals = [1, 2, 1, 64, 64, 1, 1]
    for i, (u, v, data) in enumerate(dag.edges(data=True)):
        assert data['streaming_interval'] == expected_streaming_intervals[i]

    ######## Multi path
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=128, stream=False)
    dag.add_edge(1, 2, weight=32, stream=True)
    dag.add_edge(2, 3, weight=4, stream=True)
    dag.add_edge(2, 6, weight=4, stream=True)
    dag.add_edge(3, 4, weight=1, stream=True)
    dag.add_edge(4, 5, weight=2, stream=True)
    dag.add_edge(5, 7, weight=4, stream=True)
    dag.add_edge(6, 7, weight=4, stream=True)
    dag.add_edge(7, 8, weight=4, stream=True)
    # if node 3 is buffer, the following node will be still affected by the other path
    scheduler = StreamingScheduler(dag, num_pes=1, buffer_nodes={3})
    scheduler.streaming_interval_analysis()

    expected_streaming_intervals = [1, 4, 32, 32, 128, 32, 64, 32, 32]
    for i, (u, v, data) in enumerate(dag.edges(data=True)):
        assert data['streaming_interval'] == expected_streaming_intervals[i]

    # If, instead, node 6 is buffer, then that part is isolated
    scheduler = StreamingScheduler(dag, num_pes=1, buffer_nodes={3, 6})
    scheduler.streaming_interval_analysis()
    for u, v, data in dag.edges(data=True):
        print(f"[{u}, {v}]: {data['streaming_interval']}")

    expected_streaming_intervals = [1, 4, 32, 32, 1, 1, 2, 1, 1]
    for i, (u, v, data) in enumerate(dag.edges(data=True)):
        assert data['streaming_interval'] == expected_streaming_intervals[i]

    ### Generic with buffer node in the middle
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=4, stream=False)
    dag.add_edge(0, 2, weight=4, stream=False)
    dag.add_edge(1, 3, weight=4, stream=True)
    dag.add_edge(3, 4, weight=8, stream=True)  # 3 is an upsampler
    dag.add_edge(3, 5, weight=8, stream=True)  # 5 is a buffer node
    dag.add_edge(4, 6, weight=16, stream=True)  # 4 is an upsampler
    dag.add_edge(2, 7, weight=32, stream=True)
    dag.add_edge(5, 7, weight=32, stream=False)
    dag.add_edge(7, 8, weight=4, stream=True)  # 8 is an upsampler
    dag.add_node(9, pseudo=True)
    dag.add_edge(6, 9, weight=16, stream=False)
    dag.add_edge(8, 9, weight=4, stream=False)
    buffer_nodes = {5}

    scheduler = StreamingScheduler(dag, num_pes=1, buffer_nodes=buffer_nodes)
    scheduler.streaming_interval_analysis()

    expected_streaming_intervals = [1, 1, 4, 1, 2, 2, 1, 1, 1, 8, 1]
    for i, (u, v, data) in enumerate(dag.edges(data=True)):
        assert data['streaming_interval'] == expected_streaming_intervals[i]
