# Streaming-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""

    Buffer space tests. Results are validated against Simulation

"""
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


def compute_buffer_space(dag, component=None):
    """

    :param dag: _description_
    :type dag: _type_
    """
    # Build the subgraph of the streaming block
    subg = dag.subgraph(dag.nodes() if component is None else component)
    import itertools
    import math

    # Compute all disjoint paths: do it more efficiently (.e.g, use topo order?)
    disjoint_paths = []
    for u, v in itertools.combinations(subg, 2):
        if subg.out_degree(u) > 1 and subg.in_degree(v) > 1:
            paths = list(nx.node_disjoint_paths(subg, u, v))
            if len(paths) > 1:
                disjoint_paths.append(paths)

    print("Disjoint paths: ", disjoint_paths)

    # At this point we go over the paths and we look at the latency of each of them
    global_buff_space = 1
    for dis_paths in disjoint_paths:
        latencies = []
        for path in dis_paths:
            # The latency is given by max{str_interval} + len(path) (TODO src/sink?)
            max_str_interval = 1
            src = path[0]
            for dst in path[1:]:
                max_str_interval = max(max_str_interval, subg.edges[(src, dst)]['streaming_interval'])
                src = dst

            latency = max_str_interval + len(path)

            print("Latency for path: ", path, ": ", latency)
            latencies.append(int(latency))

        # get max latency
        max_diff = 0
        for u, v in itertools.combinations(latencies, 2):
            max_diff = max(max_diff, abs(u - v))

        # TODO: divide it by input streaming interval to destination
        input_streaming_interval = subg.edges[(path[-2], path[-1])]['streaming_interval']

        #TODO: we should do the max?
        buff_space = math.ceil(max_diff / input_streaming_interval)

        print("Buff space: ", buff_space)
        global_buff_space = max(global_buff_space, buff_space)

    return global_buff_space


def test_simple_elwise():
    """
    DAGs composed only by elwises and a single streaming block. In this case, there will be no deadlocks
    but the task execution may be slowed down because of insufficient buffer space.

    """
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=32)  #pseudo root
    dag.add_edge(1, 2, weight=32, stream=True)
    dag.add_edge(2, 3, weight=32, stream=True)
    dag.add_edge(3, 4, weight=32, stream=True)
    dag.add_edge(4, 5, weight=32, stream=True)
    dag.add_edge(1, 5, weight=32, stream=True)

    num_pes = 5
    scheduler = StreamingScheduler(dag, num_pes=num_pes)
    scheduler.streaming_interval_analysis()
    # Note: with input size 4, the task 5 is associated to the same PE of another task and therefore
    # simulation would not work
    # This graph requires a buffer space of 3

    streaming_blocks = [dag.nodes()]
    pes_schedule, tasks_schedule = scheduler.schedule_dag(streaming_blocks=streaming_blocks, )

    buff_space = compute_buffer_space(dag)
    channels_buffer_spaces = dict()
    channels_buffer_spaces[(1, 5)] = buff_space
    sim = Simulation(dag,
                     tasks_schedule,
                     pes_schedule,
                     channels_capacities=channels_buffer_spaces,
                     synchronous_communication=False)
    sim.execute()

    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)

    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=32)  #pseudo root
    dag.add_edge(1, 2, weight=32, stream=True)
    dag.add_edge(2, 3, weight=32, stream=True)
    dag.add_edge(3, 4, weight=32, stream=True)
    dag.add_edge(4, 5, weight=32, stream=True)
    dag.add_edge(1, 6, weight=32, stream=True)
    dag.add_edge(6, 7, weight=32, stream=True)
    dag.add_edge(7, 8, weight=32, stream=True)
    dag.add_edge(5, 8, weight=32, stream=True)
    # This graph requires a buffer space of 2

    num_pes = 8
    scheduler = StreamingScheduler(dag, num_pes=num_pes)
    scheduler.streaming_interval_analysis()
    streaming_blocks = [dag.nodes()]
    pes_schedule, tasks_schedule = scheduler.schedule_dag(streaming_blocks=streaming_blocks, )
    # print_schedule(pes_schedule, "PE")

    buff_space = compute_buffer_space(dag)
    channels_capacities = dict()
    channels_capacities[(7, 8)] = buff_space
    sim = Simulation(dag,
                     tasks_schedule,
                     pes_schedule,
                     channels_capacities=channels_capacities,
                     synchronous_communication=False)
    sim.execute()

    # print(sim.get_task_timings())
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)

    # More than two path: add to the previous one, in this way the buffer space is again 3
    dag.add_edge(1, 9, weight=32, stream=True)
    dag.add_edge(9, 8, weight=32, stream=True)

    num_pes = 9
    scheduler = StreamingScheduler(dag, num_pes=num_pes)
    scheduler.streaming_interval_analysis()
    streaming_blocks = [dag.nodes()]

    pes_schedule, tasks_schedule = scheduler.schedule_dag(streaming_blocks=streaming_blocks, )

    buff_space = compute_buffer_space(dag)
    channels_capacities[(9, 8)] = buff_space
    sim = Simulation(dag,
                     tasks_schedule,
                     pes_schedule,
                     channels_capacities=channels_capacities,
                     synchronous_communication=False)
    sim.execute()

    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)


def test_simple_downsampler():
    """
    Test over DAGs composed by a single streaming block and elwise/downsampler nodes only
    """

    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=32)  #pseudo root
    dag.add_edge(1, 2, weight=32, stream=True)
    dag.add_edge(2, 3, weight=16, stream=True)
    dag.add_edge(3, 4, weight=8, stream=True)
    dag.add_edge(4, 5, weight=4, stream=True)
    dag.add_edge(1, 6, weight=32, stream=True)
    dag.add_edge(5, 7, weight=2, stream=True)
    dag.add_edge(6, 7, weight=2, stream=True)

    num_pes = 7
    scheduler = StreamingScheduler(dag, num_pes=num_pes)
    scheduler.streaming_interval_analysis()

    # for u, v, data in dag.edges(data=True):
    #     print(f"[{u}, {v}]: {data['streaming_interval']}")

    streaming_blocks = [dag.nodes()]
    pes_schedule, tasks_schedule = scheduler.schedule_dag(streaming_blocks=streaming_blocks, )
    print_schedule(pes_schedule, "PE")

    # we need a buffer space of 1 to be able to run at full speed (at node 7 it arrives one element every 16 clock cycles)
    buff_space = compute_buffer_space(dag)
    sim = Simulation(dag, tasks_schedule, pes_schedule, buff_space, synchronous_communication=False)
    sim.execute()

    # print(sim.get_task_timings())
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)

    # 3 paths
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=64)  #pseudo root
    dag.add_edge(1, 2, weight=64, stream=True)
    dag.add_edge(2, 3, weight=64, stream=True)
    dag.add_edge(3, 4, weight=64, stream=True)
    dag.add_edge(4, 5, weight=64, stream=True)
    dag.add_edge(5, 11, weight=4, stream=True)
    dag.add_edge(1, 6, weight=64, stream=True)
    dag.add_edge(6, 7, weight=32, stream=True)
    dag.add_edge(7, 8, weight=16, stream=True)
    dag.add_edge(8, 9, weight=8, stream=True)
    dag.add_edge(9, 11, weight=4, stream=True)
    dag.add_edge(1, 10, weight=64, stream=True)
    dag.add_edge(11, 12, weight=2, stream=True)
    dag.add_edge(10, 12, weight=2, stream=True)
    num_pes = 12
    scheduler = StreamingScheduler(dag, num_pes=num_pes)
    scheduler.streaming_interval_analysis()

    # for u, v, data in dag.edges(data=True):
    #     print(f"[{u}, {v}]: {data['streaming_interval']}")
    streaming_blocks = [dag.nodes()]
    pes_schedule, tasks_schedule = scheduler.schedule_dag(streaming_blocks=streaming_blocks, )

    print_schedule(pes_schedule, "PE")

    # we need a buffer space of 1 to be able to run at full speed (at node 7 it arrives one element every 16 clock cycles)
    buff_space = compute_buffer_space(dag)
    sim = Simulation(dag, tasks_schedule, pes_schedule, buff_space, synchronous_communication=False)
    sim.execute()

    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)


def test_simple_upsampler():
    """
    Test over DAGs composed by a single streaming block and elwise/downsampler/upsampler nodes
    """

    dag = nx.DiGraph()

    dag.add_edge(0, 1, weight=64)  #pseudo root
    dag.add_edge(1, 2, weight=64, stream=True)
    dag.add_edge(2, 3, weight=1, stream=True)
    dag.add_edge(3, 4, weight=64, stream=True)
    dag.add_edge(1, 4, weight=64, stream=True)

    num_pes = 4
    scheduler = StreamingScheduler(dag, num_pes=num_pes)
    scheduler.streaming_interval_analysis()

    for u, v, data in dag.edges(data=True):
        print(f"[{u}, {v}]: {data['streaming_interval']}")
    streaming_blocks = [dag.nodes()]
    # It needs gang schedule to avoid having two streaming tasks on the same PE
    pes_schedule, tasks_schedule = scheduler.gang_schedule(streaming_blocks)
    # print_schedule(pes_schedule, "PE")

    # TODO: the optimal buffer space is 64, but this returns 65
    buff_space = compute_buffer_space(dag)
    sim = Simulation(dag, tasks_schedule, pes_schedule, buff_space, synchronous_communication=False)
    sim.execute()

    # print(sim.get_task_timings())
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)

    # one path with downsampler/upsampler, one path with just elwise
    dag = nx.DiGraph()

    dag.add_edge(0, 1, weight=16)  #pseudo root
    dag.add_edge(1, 2, weight=16, stream=True)
    dag.add_edge(2, 3, weight=8, stream=True)
    dag.add_edge(3, 4, weight=4, stream=True)
    dag.add_edge(4, 5, weight=1, stream=True)
    dag.add_edge(1, 6, weight=16, stream=True)
    dag.add_edge(6, 7, weight=16, stream=True)
    dag.add_edge(5, 7, weight=16, stream=True)
    num_pes = 7
    scheduler = StreamingScheduler(dag, num_pes=num_pes)
    scheduler.streaming_interval_analysis()
    streaming_blocks = [dag.nodes()]
    pes_schedule, tasks_schedule = scheduler.gang_schedule(streaming_blocks)
    print_schedule(pes_schedule, "PE")

    # TODO: the optimal buffer space is 15 but this returns 18 (correctly computed according to our formula)
    buff_space = compute_buffer_space(dag)

    sim = Simulation(dag, tasks_schedule, pes_schedule, buff_space, synchronous_communication=False)
    sim.execute()

    print(sim.get_task_timings())
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag)

    # Attention, this is approximated already on its own

    dag = nx.DiGraph()

    dag.add_edge(0, 1, weight=16)  #pseudo root
    dag.add_edge(1, 2, weight=16, stream=True)
    dag.add_edge(2, 3, weight=8, stream=True)
    dag.add_edge(3, 4, weight=4, stream=True)
    dag.add_edge(4, 5, weight=2, stream=True)
    dag.add_edge(5, 6, weight=4, stream=True)
    dag.add_edge(6, 7, weight=8, stream=True)
    dag.add_edge(7, 8, weight=16, stream=True)
    dag.add_edge(1, 8, weight=16, stream=True)

    num_pes = 8
    scheduler = StreamingScheduler(dag, num_pes=num_pes)
    scheduler.streaming_interval_analysis()

    for u, v, data in dag.edges(data=True):
        print(f"[{u}, {v}]: {data['streaming_interval']}")
    streaming_blocks = [dag.nodes()]

    pes_schedule, tasks_schedule = scheduler.schedule_dag(streaming_blocks=streaming_blocks, )
    print_schedule(pes_schedule, "PE")
    # the optimal buffer space for this one is 13
    buff_space = compute_buffer_space(dag)
    sim = Simulation(dag, tasks_schedule, pes_schedule, buff_space, synchronous_communication=False)
    sim.execute()

    print(sim.get_task_timings())
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag, check_overapproximated=True)

    ## Only upsamplers

    dag = nx.DiGraph()

    dag.add_edge(0, 1, weight=4)  #pseudo root
    dag.add_edge(1, 2, weight=4, stream=True)
    dag.add_edge(2, 3, weight=8, stream=True)
    dag.add_edge(3, 4, weight=16, stream=True)
    dag.add_edge(4, 5, weight=32, stream=True)
    dag.add_edge(5, 7, weight=64, stream=True)
    dag.add_edge(1, 6, weight=4, stream=True)
    dag.add_edge(6, 7, weight=64, stream=True)

    num_pes = 7
    scheduler = StreamingScheduler(dag, num_pes=num_pes)
    scheduler.streaming_interval_analysis()

    for u, v, data in dag.edges(data=True):
        print(f"[{u}, {v}]: {data['streaming_interval']}")
    streaming_blocks = [dag.nodes()]
    pes_schedule, tasks_schedule = scheduler.schedule_dag(streaming_blocks=streaming_blocks, )

    print_schedule(pes_schedule, "PE")
    # The optimal buffer space is 3
    buff_space = compute_buffer_space(dag)
    sim = Simulation(dag, tasks_schedule, pes_schedule, buff_space, synchronous_communication=False)
    sim.execute()

    print(sim.get_task_timings())
    assert check_schedule_simulation(tasks_schedule, sim.get_task_timings(), dag, check_overapproximated=True)


def undirected_cycles():

    ##########
    # This is a quite complicated DAG (coming from cholesky), where the order of edges play a role
    # If we don't look at all the cycles (or, better, we look only at cycle basis), the computed buffer space will be wrong

    # TODO: add this to the tests

    ### Below there is a reduced version

    dag = nx.DiGraph()
    # This one breaks
    dag.add_edge(0, 1, weight=128, stream=False)
    dag.add_edge(1, 2, weight=42, stream=True)
    dag.add_edge(1, 3, weight=42, stream=True)
    dag.add_edge(1, 4, weight=42, stream=True)
    dag.add_edge(2, 5, weight=14, stream=True)
    dag.add_edge(2, 7, weight=14, stream=True)
    dag.add_edge(2, 9, weight=14, stream=True)
    dag.add_edge(3, 7, weight=14, stream=True)
    dag.add_edge(3, 11, weight=14, stream=True)
    dag.add_edge(3, 14, weight=14, stream=True)
    dag.add_edge(4, 9, weight=14, stream=True)
    dag.add_edge(4, 14, weight=14, stream=True)
    dag.add_edge(4, 16, weight=14, stream=True)
    dag.add_edge(5, 6, weight=3, stream=True)
    dag.add_edge(6, 8, weight=12, stream=True)
    dag.add_edge(6, 10, weight=12, stream=True)
    dag.add_edge(7, 8, weight=12, stream=True)
    dag.add_edge(8, 12, weight=3, stream=True)
    dag.add_edge(8, 15, weight=3, stream=True)
    dag.add_edge(9, 10, weight=12, stream=True)
    dag.add_edge(10, 15, weight=3, stream=True)
    dag.add_edge(10, 17, weight=3, stream=False)
    dag.add_edge(11, 12, weight=3, stream=True)
    dag.add_edge(12, 13, weight=9, stream=True)
    dag.add_edge(13, 17, weight=3, stream=False)
    dag.add_edge(14, 15, weight=3, stream=True)
    dag.add_edge(15, 17, weight=3, stream=False)
    dag.add_edge(16, 17, weight=3, stream=False)
    dag.add_node(0)
    dag.add_node(17, pseudo=True)

    # this one works
    dag2 = nx.DiGraph()
    dag2.add_edge(0, 1, weight=128, stream=False)
    dag2.add_edge(1, 2, weight=42, stream=True)
    dag2.add_edge(1, 3, weight=42, stream=True)
    dag2.add_edge(1, 4, weight=42, stream=True)
    dag2.add_edge(4, 9, weight=14, stream=True)
    dag2.add_edge(4, 14, weight=14, stream=True)
    dag2.add_edge(4, 16, weight=14, stream=True)
    dag2.add_edge(16, 17, weight=3, stream=False)
    dag2.add_edge(3, 7, weight=14, stream=True)
    dag2.add_edge(3, 11, weight=14, stream=True)
    dag2.add_edge(3, 14, weight=14, stream=True)
    dag2.add_edge(14, 15, weight=3, stream=True)
    dag2.add_edge(11, 12, weight=3, stream=True)
    dag2.add_edge(2, 5, weight=14, stream=True)
    dag2.add_edge(2, 7, weight=14, stream=True)
    dag2.add_edge(2, 9, weight=14, stream=True)
    dag2.add_edge(9, 10, weight=12, stream=True)
    dag2.add_edge(7, 8, weight=12, stream=True)
    dag2.add_edge(5, 6, weight=3, stream=True)
    dag2.add_edge(6, 8, weight=12, stream=True)
    dag2.add_edge(6, 10, weight=12, stream=True)
    dag2.add_edge(10, 15, weight=3, stream=True)
    dag2.add_edge(10, 17, weight=3, stream=False)
    dag2.add_edge(8, 12, weight=3, stream=True)
    dag2.add_edge(8, 15, weight=3, stream=True)
    dag2.add_edge(15, 17, weight=3, stream=False)
    dag2.add_edge(12, 13, weight=9, stream=True)
    dag2.add_edge(13, 17, weight=3, stream=False)
    dag2.add_node(0)
    dag2.add_node(17, pseudo=True)

    dag = nx.DiGraph()
    # Reduced one, test with 11 PEs, the problem is the node 8
    # and the edge 7->8 that needs enough buffer space, and it was not taken into account by cycle basis
    # (also 9->10 needs buffer space). A buffer space of 2 is enough, but the compute_buffer_space3 may
    # think it needs more

    dag.add_edge(0, 1, weight=128, stream=False)
    dag.add_edge(1, 2, weight=42, stream=True)
    dag.add_edge(1, 3, weight=42, stream=True)
    dag.add_edge(1, 4, weight=42, stream=True)
    dag.add_edge(2, 5, weight=14, stream=True)

    dag.add_edge(3, 7, weight=14, stream=True)
    dag.add_edge(4, 9, weight=14, stream=True)
    dag.add_edge(5, 6, weight=3, stream=True)
    dag.add_edge(6, 8, weight=12, stream=True)
    dag.add_edge(6, 10, weight=12, stream=True)
    dag.add_edge(7, 8, weight=12, stream=True)
    dag.add_edge(8, 11, weight=3, stream=True)
    dag.add_edge(9, 10, weight=12, stream=True)
    dag.add_edge(10, 11, weight=3, stream=True)
    dag.add_edge(10, 12, weight=3, stream=False)
    dag.add_edge(11, 12, weight=3, stream=False)
    dag.add_node(0)
    dag.add_node(12, pseudo=True)

    # Reduced graph coming from resnet
    # RIDOTTO ULTERIORMENTE
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=48, stream=False)
    # dag.add_edge(0, 1, weight=768, stream=False)
    # dag.add_edge(1, 2, weight=192, stream=True)
    dag.add_edge(1, 2, weight=24, stream=True)
    dag.add_edge(2, 3, weight=24, stream=True)
    dag.add_edge(2, 8, weight=24, stream=True)
    dag.add_edge(3, 4, weight=6, stream=True)
    dag.add_edge(4, 5, weight=4, stream=True)
    dag.add_edge(5, 6, weight=4, stream=True)
    dag.add_edge(6, 7, weight=5, stream=True)
    dag.add_edge(7, 8, weight=24, stream=True)
    dag.add_edge(8, 9, weight=6, stream=True)
    dag.add_edge(9, 10, weight=4, stream=True)
    dag.add_edge(9, 11, weight=4, stream=True)
    dag.add_edge(11, 12, weight=2, stream=False)
    dag.add_edge(10, 12, weight=1, stream=False)
    dag.add_node(0)
    dag.add_node(12, pseudo=True)