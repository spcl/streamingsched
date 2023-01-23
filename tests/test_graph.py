# S-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    Minimal Tests on graph utils
"""

from sched.streaming_sched import StreamingScheduler, ScheduleEvent
from sched.utils import print_schedule, build_tasks_schedule_from_pes_schedule
from utils.graph import get_undirected_cycles
from utils.visualize import visualize_dag, show_schedule_gantt_chart
import networkx as nx
import logging
import pytest
from unittest import TestCase


def test_undirected_cycles():

    # Single undirected cycle
    dag = nx.DiGraph()
    dag.add_node(0)  # pseudo source
    dag.add_edge(0, 1, weight=32)
    dag.add_edge(1, 2, weight=32, stream=True)
    dag.add_edge(2, 3, weight=4, stream=True)
    dag.add_edge(3, 4, weight=2, stream=True)
    dag.add_edge(4, 5, weight=32, stream=True)
    dag.add_edge(1, 5, weight=32, stream=True)
    cycles = get_undirected_cycles(dag)
    expected = [{1, 5, 4, 3, 2}]
    assert expected == cycles

    # two separated cycles, single component
    dag = nx.DiGraph()
    dag.add_node(0)  # pseudo source
    dag.add_edge(0, 1)
    dag.add_edge(1, 2)
    dag.add_edge(2, 3)
    dag.add_edge(3, 4)
    dag.add_edge(2, 4)
    dag.add_edge(1, 5)
    dag.add_edge(5, 6)
    dag.add_edge(6, 7)
    dag.add_edge(5, 7)
    expected = [{5, 6, 7}, {2, 3, 4}]
    cycles = get_undirected_cycles(dag)
    assert expected == cycles

    # then we merge the two cycles
    dag.add_edge(4, 7)
    expected = [{2, 3, 4, 1, 5, 6, 7}]
    cycles = get_undirected_cycles(dag)
    assert expected == cycles

    # three paths
    dag = nx.DiGraph()

    dag.add_edge(0, 1, weight=16, stream=False)
    dag.add_edge(1, 2, weight=16, stream=True)
    dag.add_edge(2, 3, weight=16, stream=True)
    dag.add_edge(3, 4, weight=16, stream=True)
    dag.add_edge(4, 6, weight=16, stream=True)
    dag.add_edge(1, 6, weight=16, stream=True)
    dag.add_edge(1, 5, weight=16, stream=True)
    dag.add_edge(5, 6, weight=16, stream=True)
    expected = [{1, 2, 3, 4, 5, 6}]
    # all of them in the same cycle
    cycles = get_undirected_cycles(dag)
    assert expected == cycles

    # nested cycles
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=32, stream=False)  # pseudo source
    dag.add_edge(1, 2, weight=32, stream=True)
    dag.add_edge(2, 3, weight=2, stream=True)
    dag.add_edge(3, 4, weight=32, stream=True)
    dag.add_edge(1, 4, weight=32, stream=True)
    dag.add_edge(1, 5, weight=32, stream=True)
    dag.add_edge(4, 6, weight=2, stream=True)
    dag.add_edge(6, 7, weight=32, stream=True)
    dag.add_edge(3, 7, weight=32, stream=True)
    dag.add_edge(5, 7, weight=32, stream=True)
    cycles = get_undirected_cycles(dag)
    # we return all the nodes in the largest one
    expected = [{1, 2, 3, 4, 5, 6, 7}]
    assert expected == cycles

    # separate components
    dag = nx.DiGraph()
    dag.add_edge(2, 3, weight=1)
    dag.add_edge(3, 4, weight=1)
    dag.add_edge(2, 4, weight=1)
    dag.add_edge(1, 5, weight=1)
    dag.add_edge(5, 6, weight=1)
    dag.add_edge(6, 7, weight=1)
    dag.add_edge(5, 7, weight=1)
    cycles = get_undirected_cycles(dag)
    expected = [{2, 3, 4}, {5, 6, 7}]
    assert cycles == expected


def test_medium_undirected_cycles():
    '''
        Cycles of medium size
    '''
    dag = nx.DiGraph()
    dag.add_edge(0, 1, weight=16, stream=False)
    dag.add_edge(1, 2, weight=16, stream=True)
    dag.add_edge(1, 3, weight=16, stream=True)
    dag.add_edge(1, 4, weight=16, stream=True)
    dag.add_edge(4, 7, weight=16, stream=True)
    dag.add_edge(3, 6, weight=16, stream=True)
    dag.add_edge(2, 5, weight=16, stream=False)
    dag.add_edge(5, 6, weight=16, stream=True)
    dag.add_edge(5, 7, weight=16, stream=True)
    dag.add_edge(7, 9, weight=64, stream=True)
    dag.add_edge(6, 8, weight=64, stream=True)
    dag.add_edge(8, 9, weight=64, stream=True)

    # All the nodes in this graph (except 0) are part of an undirected cycle
    # This shows how we can not stop marking as soon as we found a marked node,
    # as there can be other unmarked node before reaching the common ancestor
    # (In this case, we have a first backward edge 9->7, then one 3->1 but some nodes
    # in the middle are left unmarked)
    cycles = get_undirected_cycles(dag)
    expected = [{1, 2, 3, 4, 5, 6, 7, 8, 9}]
    assert cycles == expected
