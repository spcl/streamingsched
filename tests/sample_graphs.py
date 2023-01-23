'''
    Contains a collection of sample graphs used for testing

'''

import networkx as nx
import numpy as np


def build_dag_4(num_pes=3, lat=0, same_weights=False):
    '''
    Builds a Simple graph with 4 nodes, in a rhomboid shape, and 3 PEs (by default)
    PEs have different II. Lat is 0 by default, for all of them.
    :return: dag, II and Lat
    '''
    dag = nx.DiGraph()

    if same_weights:
        dag.add_edge(0, 1, weight=5)
        dag.add_edge(0, 2, weight=7)
        dag.add_edge(1, 3, weight=4)
        dag.add_edge(2, 3, weight=4)
        return dag
    else:
        dag.add_edge(0, 1, weight=5)
        dag.add_edge(0, 2, weight=7)
        dag.add_edge(1, 3, weight=4)
        dag.add_edge(2, 3, weight=3)
        num_tasks = len(dag.nodes())

        #########
        # Build II matrix
        #########
        II = [None] * num_tasks

        # 0 is the (pseudo) root
        II[0] = np.zeros((num_pes, 1))
        # Build matrix PExInputs for task 1
        II[1] = np.zeros((num_pes, 1))
        II[1][0][0] = 1
        II[1][1][0] = 2
        II[1][2][0] = 3

        # Task 2
        II[2] = np.zeros((num_pes, 1))
        II[2][0][0] = 3
        II[2][1][0] = 2
        II[2][2][0] = 4

        # Task 3
        II[3] = np.ones((num_pes, 2))

        ### Build Lat matrix
        Lat = np.full((num_tasks, num_pes), lat)

        return dag, II, Lat


def build_dag_8(num_pes=3, lat=0, same_weights=False):
    '''
    Builds a dag with 8 tasks.
    If same_weights is True, just build the graph and all edges incident to a node have the same weight
    By default, PEs have all II=1 and latency is 0, number of PEs is 3
    :return: dag, II and Lat
    '''
    dag = nx.DiGraph()
    if same_weights:
        dag.add_edge(0, 1, weight=4)
        dag.add_edge(0, 2, weight=3)
        dag.add_edge(0, 3, weight=12)
        dag.add_edge(1, 5, weight=11)
        dag.add_edge(1, 6, weight=11)
        dag.add_edge(2, 4, weight=5)
        dag.add_edge(3, 5, weight=11)
        dag.add_edge(3, 6, weight=11)
        dag.add_edge(4, 7, weight=3)
        dag.add_edge(5, 7, weight=3)
        dag.add_edge(6, 7, weight=3)
        return dag
    else:
        dag.add_edge(0, 1, weight=4)
        dag.add_edge(0, 2, weight=3)
        dag.add_edge(0, 3, weight=12)
        dag.add_edge(1, 5, weight=6)
        dag.add_edge(1, 6, weight=11)
        dag.add_edge(2, 4, weight=5)
        dag.add_edge(3, 5, weight=3)
        dag.add_edge(3, 6, weight=2)
        dag.add_edge(4, 7, weight=3)
        dag.add_edge(5, 7, weight=4)
        dag.add_edge(6, 7, weight=5)
        num_tasks = len(dag.nodes())

        #########
        # Build II matrix: everything has II =1
        #########
        # First five tasks have one input
        II = [np.ones((num_pes, 1))] * 5

        # Tasks 5 and 6 have two inputs
        II.append(np.ones((num_pes, 2)))
        II.append(np.ones((num_pes, 2)))

        # Task 7 has three inputs
        II.append(np.ones((num_pes, 3)))

        ### Build Lat matrix
        Lat = np.full((num_tasks, num_pes), lat)

        return dag, II, Lat