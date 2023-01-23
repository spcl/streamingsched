# S-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Tests for evaluating the simulation under controlled scenarios.
    We consider both backpressure and not.

'''
import packaging
from sched.streaming_sched import StreamingScheduler, ScheduleEvent
from sched.utils import print_schedule, build_tasks_schedule_from_pes_schedule
from utils.visualize import visualize_dag, show_schedule_gantt_chart
from sample_graphs import *
import networkx as nx
import numpy as np
from sched.simulate import Simulation
import logging
import pytest
from unittest import TestCase
from sched.utils import check_schedule_simulation


def test_simple():
    '''

    Tests over simple combination of nodes
    '''

    # Elwise
    W = 8
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=W)
    dag.add_edge(1, 2, weight=W)  # non pseudo sink

    scheduler = StreamingScheduler(dag, num_pes=2)
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    sim = Simulation(dag, tasks_schedule, pes_schedule, 1)
    sim.execute()
    # print(sim.get_task_timings())
    # print_schedule(pes_schedule, "pe")
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)

    # Streaming edge
    dag[1][2]['stream'] = True
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    sim = Simulation(dag, tasks_schedule, pes_schedule, 1)
    sim.execute()
    print(sim.get_task_timings())
    print_schedule(pes_schedule, "pe")
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)

    ## Downsampler, no streaming: the second node will finish at time 10

    dag[1][2]['weight'] = 2
    dag[1][2]['stream'] = False
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    sim = Simulation(dag, tasks_schedule, pes_schedule, 1)
    print_schedule(pes_schedule, "pe")
    sim.execute()
    print(sim.get_task_timings())
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)

    # Streaming edge, the second node will finish at time 9
    dag[1][2]['stream'] = True
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    sim = Simulation(dag, tasks_schedule, pes_schedule, 1)
    sim.execute()
    print(sim.get_task_timings())
    print_schedule(pes_schedule, "pe")
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)

    # Non integer downsampling ratio
    dag[1][2]['weight'] = 3
    dag[1][2]['stream'] = True
    scheduler = StreamingScheduler(dag, num_pes=2)
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    sim = Simulation(dag, tasks_schedule, pes_schedule, 1)
    print_schedule(pes_schedule, "pe")
    sim.execute()

    ## Upsampler, no streaming

    dag[1][2]['weight'] = 16
    dag[1][2]['stream'] = False
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    sim = Simulation(dag, tasks_schedule, pes_schedule, 1)
    print_schedule(pes_schedule, "pe")
    sim.execute()
    print(sim.get_task_timings())
    # assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)

    # # Streaming
    dag[1][2]['stream'] = True
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    sim = Simulation(dag, tasks_schedule, pes_schedule, 1)
    print_schedule(pes_schedule, "pe")
    sim.execute()
    print(sim.get_task_timings())
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)


def test_chain():
    '''
    Tests simulation over a streaming chain of tasks
    Most of them are borrowed from test_backpressure (with different volume)
    '''

    N = 4
    M = 8
    dag = nx.DiGraph()

    ### All Element-Wise

    # pseudoroot
    dag.add_edge(0, 1, weight=M)
    for i in range(N - 1):
        dag.add_edge(i + 1, i + 2, weight=M, stream=True)

    # add a pseudo sink node
    dag.add_node(N + 1, pseudo=True)
    dag.add_edge(N, N + 1, weight=M)

    num_pes = N
    num_tasks = N + 1 + 1  # considering the pseudoroot and pseudo sink

    scheduler = StreamingScheduler(dag, num_pes=num_pes)
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()

    sim = Simulation(dag, tasks_schedule, pes_schedule, 1)
    sim.execute()
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)

    ##### Downsampler and upsampler
    dag[1][2]['weight'] = 2
    dag[2][3]['weight'] = 8

    scheduler.streaming_interval_analysis()

    pes_schedule, tasks_schedule = scheduler.schedule_dag()

    sim = Simulation(dag, tasks_schedule, pes_schedule, 1)
    sim.execute()
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)

    # Let's have an upsampler (NOTE that 0 is the pseudo-root)
    # 0 -- (8) --> 1 -- (16) --> 2 -- (32) --> 3 -- (16) --> 4
    # This example shows how we are currently modeling synchronous communication
    dag[1][2]['weight'] = 16
    dag[2][3]['weight'] = 32

    scheduler.streaming_interval_analysis()

    pes_schedule, tasks_schedule = scheduler.schedule_dag()

    sim = Simulation(dag, tasks_schedule, pes_schedule, 1, synchronous_communication=True)
    sim.execute()
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)

    # put a non streaming edge
    dag[1][2]['stream'] = False

    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()

    sim = Simulation(dag, tasks_schedule, pes_schedule, 1)
    sim.execute()
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)

    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=128, stream=False)
    dag.add_edge(1, 2, weight=32, stream=True)
    dag.add_edge(2, 3, weight=32, stream=True)
    dag.add_edge(3, 4, weight=8, stream=True)
    dag.add_edge(4, 5, weight=8, stream=True)

    scheduler = StreamingScheduler(dag, num_pes=5)

    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()

    sim = Simulation(dag, tasks_schedule, pes_schedule, 1)
    sim.execute()
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)

    # Chain with non-integer downsampling ratio. Here the problem is mainly floating point precision
    # that may result in wrong comparison

    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=8, stream=False)
    dag.add_edge(1, 2, weight=8, stream=True)
    dag.add_edge(2, 3, weight=16, stream=True)
    dag.add_edge(3, 4, weight=11, stream=True)
    dag.add_edge(4, 5, weight=4, stream=False)

    scheduler = StreamingScheduler(dag, num_pes=4)
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    sim = Simulation(dag, tasks_schedule, pes_schedule, 1, synchronous_communication=True)
    sim.execute()
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag, check_overapproximated=True)

    # Non-integer upsampling ratio
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=18, stream=False)
    dag.add_edge(1, 2, weight=24, stream=True)
    dag.add_edge(2, 3, weight=48, stream=True)
    dag.add_edge(3, 4, weight=48, stream=True)

    scheduler = StreamingScheduler(dag, num_pes=4)
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    sim = Simulation(dag, tasks_schedule, pes_schedule, 1, synchronous_communication=True)
    sim.execute()
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)


def test_approximate():
    ''' 
        Test cases where the scheduling is pessimistic and produces an overapproximated
        makespan. 

        In this case we want to check that the computed schedule is actually an overapproximation
        than the simulated one

    '''

    #####
    # Chain of N-1 elwise-downsampler tasks with an up sampler at the end. According to streaming
    # interval analysis the upsampler will affect all streaming intervals. However, the other nodes
    # will be affected by the backpressure only from one point on. For example, the first
    # task can produce up to N-1 elements without any backpressure.

    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=16, stream=False)
    dag.add_edge(1, 2, weight=16, stream=True)
    dag.add_edge(2, 3, weight=2, stream=True)
    dag.add_edge(3, 4, weight=2, stream=True)
    dag.add_edge(4, 5, weight=32, stream=True)

    scheduler = StreamingScheduler(dag, num_pes=5)
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()

    sim = Simulation(dag, tasks_schedule, pes_schedule, 1)
    sim.execute()

    # print(sim.get_task_timings())
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag, check_overapproximated=True)


def test_multi_input():
    '''
    Basic topology with symmetric paths
           ┌────► 3 ──────5───────
      0─►1-│                      ▼
           └────► 2 ────► 4 ────► 6
    '''

    dag = nx.DiGraph()
    dag.add_node(0)
    dag.add_node(1)
    dag.add_node(2)
    dag.add_node(3)
    dag.add_node(4)
    dag.add_node(5)
    dag.add_edge(0, 1, weight=4)  #pseudo root
    dag.add_edge(1, 2, weight=4, stream=True)
    dag.add_edge(1, 3, weight=4, stream=True)
    dag.add_edge(2, 4, weight=4, stream=True)
    dag.add_edge(3, 5, weight=4, stream=True)
    dag.add_edge(4, 6, weight=4, stream=True)
    dag.add_edge(5, 6, weight=4, stream=True)

    # no backpressure
    num_pes = 6
    scheduler = StreamingScheduler(dag, num_pes=num_pes)

    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    print_schedule(pes_schedule, "PE")

    sim = Simulation(dag, tasks_schedule, pes_schedule, 1)
    sim.execute()

    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)

    # TODO: backpressure
    # Now, node 2 is an upsampler (and node 4) a downsampler. So it will slow-down node 1 and also node 3, 5
    # This is another case where the current scheduling strategy does not matches the simulation:
    # - according to the schedule, task 5 starts at time 2 and finishes at time 9
    # - according to the simulation, it finishes instead at time 10.
    # This is due to the synchronous communication mode: task 5 will produce the first result at time
    #   3, while task 4 will produce at time 4. Therefore they need to synch up, and the execution of
    #   task 5 would follow the execution of task 4

    dag[2][4]['weight'] = 8
    scheduler = StreamingScheduler(dag, num_pes=num_pes)
    scheduler.streaming_interval_analysis()
    for u, v, data in dag.edges(data=True):
        print(f"[{u}, {v}]: {data['streaming_interval']}")

    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    print_schedule(pes_schedule, "PE")

    sim = Simulation(dag, tasks_schedule, pes_schedule, 1)
    sim.execute()

    print(sim.get_task_timings())
    # assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)


def test_misc():
    '''
        Misc tests, particular cases that were causing issues.
        Only using the streaming inteval analysis
    '''

    # non integer upsampling ratios

    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=16)
    dag.add_edge(0, 2, weight=4)
    dag.add_edge(1, 3, weight=7, stream=False)
    dag.add_edge(2, 3, weight=7, stream=False)
    dag.add_edge(3, 4, weight=7, stream=True)

    num_pes = 4
    scheduler = StreamingScheduler(dag, num_pes=num_pes)

    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    print_schedule(pes_schedule, "pe")
    sim = Simulation(dag, tasks_schedule, pes_schedule, 1, synchronous_communication=False)
    sim.execute(print_time=True)
    print(sim.get_task_timings())
    for u, v, data in dag.edges(data=True):
        print(f"[{u}, {v}]: {data['streaming_interval']}")
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)

    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=16)
    dag.add_edge(0, 2, weight=4)
    dag.add_edge(1, 3, weight=14, stream=False)
    dag.add_edge(2, 3, weight=14, stream=False)
    dag.add_edge(3, 4, weight=14, stream=True)

    num_pes = 4
    scheduler = StreamingScheduler(dag, num_pes=num_pes)
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    sim = Simulation(dag, tasks_schedule, pes_schedule, 1)
    sim.execute(print_time=True)
    print(sim.get_task_timings())
    print_schedule(pes_schedule, "pe")

    # In this case, one of the tasks (task 2) finishes slightly later with simulation because of
    # the non integer upsampling ratio (and additional reads)
    # assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)

    # upstream and stream
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=16)
    dag.add_edge(1, 2, weight=16, stream=True)
    dag.add_edge(2, 3, weight=8, stream=True)
    dag.add_edge(3, 4, weight=23, stream=False)

    num_pes = 4
    scheduler = StreamingScheduler(dag, num_pes=num_pes)
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    sim = Simulation(dag, tasks_schedule, pes_schedule, 1)
    sim.execute(print_time=True)
    print_schedule(pes_schedule, "pe")
    print(sim.get_task_timings())
    # the upsampler is at the end
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag, check_overapproximated=True)

    # non integer downsampling ratios

    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=8)
    dag.add_edge(1, 2, weight=16, stream=True)
    dag.add_edge(2, 3, weight=9, stream=True)
    dag.add_edge(3, 4, weight=4)

    num_pes = 4
    scheduler = StreamingScheduler(dag, num_pes=num_pes)

    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    print_schedule(pes_schedule, "pe")
    sim = Simulation(dag, tasks_schedule, pes_schedule, 1, synchronous_communication=False)
    sim.execute(print_time=True)
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag, check_overapproximated=True)

    # TODO: this seems okish
    dag = nx.DiGraph()
    dag.add_edge(0, 2, weight=8, stream=False)
    dag.add_edge(0, 1, weight=8, stream=False)
    dag.add_edge(0, 1, weight=8, stream=False)
    dag.add_edge(1, 2, weight=8, stream=True)
    dag.add_edge(1, 4, weight=8, stream=True)
    dag.add_edge(4, 5, weight=9, stream=True)
    dag.add_edge(0, 2, weight=8, stream=False)
    dag.add_edge(2, 3, weight=16, stream=True)
    dag.add_edge(0, 3, weight=16, stream=False)
    dag.add_edge(3, 5, weight=9, stream=True)
    dag.add_edge(5, 6, weight=4, stream=False)

    # THIS DOES NOT WORK WITH Synchronous communication because of asymmetric path

    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=128)
    dag.add_edge(1, 2, weight=128, stream=True)
    dag.add_edge(1, 3, weight=128, stream=True)
    dag.add_edge(2, 4, weight=64, stream=True)
    dag.add_edge(2, 6, weight=64, stream=True)
    dag.add_edge(4, 5, weight=21, stream=True)
    dag.add_edge(5, 7, weight=63, stream=True)
    dag.add_edge(6, 7, weight=63, stream=True)
    dag.add_edge(3, 6, weight=64, stream=True)
    # dag.add_edge(3, 8, weight=64, stream=True)
    # dag.add_edge(7, 9, weight=189, stream=False)
    # dag.add_edge(8, 9, weight=189, stream=False)
    # visualize_dag(dag)
    num_pes = 8
    scheduler = StreamingScheduler(dag, num_pes=num_pes)

    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    print_schedule(pes_schedule, "pe")
    sim = Simulation(dag, tasks_schedule, pes_schedule, 1, synchronous_communication=False)
    sim.execute(print_time=True)
    print(sim.get_task_timings())

    # Cholesky DAG
    # dag = nx.DiGraph()
    # dag.add_edge(0, 1, weight=128)
    # dag.add_edge(1, 2, weight=128, stream=True)
    # dag.add_edge(1, 3, weight=128, stream=True)
    # dag.add_edge(2, 4, weight=64, stream=True)
    # dag.add_edge(2, 6, weight=64, stream=True)
    # dag.add_edge(4, 5, weight=21, stream=True)
    # dag.add_edge(5, 7, weight=63, stream=True)
    # dag.add_edge(6, 7, weight=63, stream=True)
    # dag.add_edge(3, 6, weight=64, stream=True)
    # dag.add_edge(3, 8, weight=64, stream=True)
    # dag.add_edge(7, 9, weight=189, stream=False)
    # dag.add_edge(8, 9, weight=189, stream=False)

    # TODO: this is good for upsampling with non integer ratio
    # dag = nx.DiGraph()
    # dag.add_edge(0, 1, weight=3)
    # dag.add_edge(0, 2, weight=1)
    # dag.add_edge(1, 3, weight=3, stream=True)
    # dag.add_edge(2, 3, weight=3, stream=True)
    # dag.add_edge(3, 4, weight=8)


def test_asymmetric_paths():
    # See test_buffer_space for more complete tests
    # Here there are just a few that are relevant to the simulation itself
    # (and require buffer space 1)

    dag = nx.DiGraph()
    # This case was solved by letting tasks flush all data ready to send
    dag.add_node(7, pseudo=True)
    dag.add_edge(0, 1, weight=256, stream=False)
    dag.add_edge(1, 2, weight=256, stream=True)
    dag.add_edge(1, 4, weight=256, stream=True)
    dag.add_edge(1, 6, weight=256, stream=True)
    dag.add_edge(6, 7, weight=6144, stream=False)
    dag.add_edge(0, 4, weight=256, stream=False)
    dag.add_edge(4, 5, weight=6144, stream=True)
    dag.add_edge(0, 2, weight=256, stream=False)
    dag.add_edge(2, 3, weight=3072, stream=True)
    dag.add_edge(0, 3, weight=3072, stream=False)
    dag.add_edge(3, 5, weight=6144, stream=True)
    dag.add_edge(3, 7, weight=6144, stream=False)
    dag.add_edge(0, 5, weight=6144, stream=False)
    dag.add_edge(5, 7, weight=3072, stream=False)

    num_pes = 8
    scheduler = StreamingScheduler(dag, num_pes=num_pes)
    scheduler.streaming_interval_analysis()
    pes_schedule, tasks_schedule = scheduler.schedule_dag()
    print_schedule(pes_schedule, "pe")
    sim = Simulation(dag, tasks_schedule, pes_schedule, 1, synchronous_communication=False)
    sim.execute(print_time=True)
    print(sim.get_task_timings())
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag, check_overapproximated=True)


def test_buffer_nodes():
    ''' 
        Tests with buffer nodes
    '''

    ## chain
    dag = nx.DiGraph()
    dag.add_node(0)
    dag.add_edge(0, 1, weight=128, stream=False)
    dag.add_edge(1, 2, weight=64, stream=True)
    dag.add_edge(2, 3, weight=32, stream=False)
    dag.add_edge(3, 4, weight=16, stream=True)

    #Let's assume that node 2 is buffer node
    scheduler = StreamingScheduler(dag, num_pes=4, buffer_nodes={2})
    scheduler.streaming_interval_analysis()
    blocks = scheduler.get_streaming_blocks()
    pes_schedule, tasks_schedule = scheduler.gang_schedule(blocks)
    sim = Simulation(dag, tasks_schedule, pes_schedule, 1, synchronous_communication=False, buffer_nodes={2})
    sim.execute()
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)

    # Chain with buffer node right after source node. The buffer node is ready to go at time 0 and finishes at time 0
    # (it is like a source)
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=128, stream=False)
    dag.add_edge(1, 2, weight=42, stream=False)
    dag.add_edge(2, 3, weight=14, stream=True)
    dag.add_edge(3, 4, weight=3, stream=True)
    dag.add_edge(4, 5, weight=12, stream=True)
    dag.add_edge(5, 6, weight=3, stream=True)
    dag.add_edge(6, 7, weight=9, stream=True)
    dag.add_edge(7, 8, weight=3, stream=True)
    dag.add_edge(8, 9, weight=3, stream=False)

    #Let's assume that node 2 is buffer node
    scheduler = StreamingScheduler(dag, num_pes=8, buffer_nodes={1})
    scheduler.streaming_interval_analysis()
    blocks = scheduler.get_streaming_blocks()
    pes_schedule, tasks_schedule = scheduler.gang_schedule(blocks)
    sim = Simulation(dag, tasks_schedule, pes_schedule, 1, synchronous_communication=False, buffer_nodes={1})
    sim.execute()
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag, check_overapproximated=True)

    ######## Multi path
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=128, stream=False)
    dag.add_edge(1, 2, weight=32, stream=True)
    dag.add_edge(2, 3, weight=4, stream=True)
    dag.add_edge(2, 6, weight=4, stream=True)
    dag.add_edge(3, 4, weight=1, stream=False)
    dag.add_edge(4, 5, weight=2, stream=True)
    dag.add_edge(5, 7, weight=4, stream=True)
    dag.add_edge(6, 7, weight=4, stream=True)
    dag.add_edge(7, 8, weight=4, stream=True)
    # if node 3 is buffer, the following node will be still affected by the other path
    scheduler = StreamingScheduler(dag, num_pes=8, buffer_nodes={3})
    scheduler.streaming_interval_analysis()
    blocks = scheduler.get_streaming_blocks()

    pes_schedule, tasks_schedule = scheduler.gang_schedule(blocks)

    # we need to use some buffer space to avoid deadlocks
    channels_capacities = dict()
    channels_capacities[2, 6] = 4
    sim = Simulation(dag,
                     tasks_schedule,
                     pes_schedule,
                     1,
                     synchronous_communication=False,
                     channels_capacities=channels_capacities,
                     buffer_nodes={3})
    sim.execute(print_time=True)

    # This is a situation where node  2 finishes much earlier than node 7 starts, and therefore the schedule is incorrect

    # If, instead, node 6 is buffer, then that part is isolated
    dag[6][7]['stream'] = False
    scheduler = StreamingScheduler(dag, num_pes=4, buffer_nodes={3, 6})
    scheduler.streaming_interval_analysis()
    blocks = scheduler.get_streaming_blocks()
    pes_schedule, tasks_schedule = scheduler.gang_schedule(blocks)

    # we need to use some buffer space to avoid deadlocks
    sim = Simulation(dag,
                     tasks_schedule,
                     pes_schedule,
                     1,
                     synchronous_communication=False,
                     channels_capacities=channels_capacities,
                     buffer_nodes={3, 6})
    sim.execute(print_time=True)
    # print_schedule(pes_schedule, "pe")
    # print(sim.get_task_timings())
    # According to the scheduling, task 4 takes 1 additional unit of time to complete
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag, check_overapproximated=True)
