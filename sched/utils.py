'''
Utility functions
'''

from typing import Set
import networkx as nx
import numpy as np


def print_schedule(schedule, type: str):
    # Utility function to print a schedule in a nice way
    print(f"----------------  {type} schedule ----------------")
    for k, v in sorted(schedule.items()):
        print(f"{type} {k}:")
        if isinstance(v, list):
            for i in v:
                print("\t", i)
        else:
            print("\t", v)


def build_tasks_schedule_from_pes_schedule(pes_schedule):
    # Utility function to build the tasks schedule from the given pes schedule
    tasks_sched = dict()
    for k, v in pes_schedule.items():
        for sched_ev in v:
            tasks_sched[sched_ev.task] = sched_ev
    return tasks_sched


def check_schedule_simulation(tasks_schedules, task_simulated_timings, dag, check_overapproximated=False):
    '''
    Checks that the result of the simulation matches the computed schedule
    :param tasks_schedules:
    :param task_simulated_timings:
    :param dag:
    :param check_overapproximated: if True, check that the computed schedule is an overapproximation
        of the simulated one (all simulated tasks should take less or equal time)
    :return: True or False
    '''

    ok = True

    for task, sched in tasks_schedules.items():
        # skip pseudo root
        if task == 0:
            continue

        # skip pseudo sink
        if task == len(tasks_schedules) - 1 and 'pseudo' in dag.nodes()[task] and dag.nodes()[task]['pseudo']:
            continue

        if check_overapproximated:
            ok = (task_simulated_timings[task][1] - task_simulated_timings[task][0]) <= (
                sched.end_t - sched.start_t) and sched.start_t >= task_simulated_timings[task][0]
            if not ok:
                import pdb
                pdb.set_trace()
        else:
            ok = sched.start_t == task_simulated_timings[task][0] and sched.end_t == task_simulated_timings[task][1]

        if not ok:
            break
    return ok


def build_W_matrix_HEFT(dag: nx.DiGraph, pseudo_source_node: int, pseudo_sink: int, buffer_nodes: Set, num_pes: int):
    '''
    Builds a computation matrix W to be used with HEFT
    Computation cost are equivalent to the execution time in isolation (that is the max of input and output volume)
    Buffer node and pseudo nodes have computation cost zero.
    
    :param dag:
    :param pseudo source_node: source node of the dag (pseudo source node)
    :param pseudo_sink:
    :param buffer_nodes: set containing the buffer nodes
    :num_pes:
    :return: the W matrix needed to run HEFT
    '''

    num_tasks = len(dag.nodes())

    comp_array = np.zeros((num_tasks))

    for task in dag.nodes():
        if task == pseudo_source_node or task == pseudo_sink or task in buffer_nodes:
            comp_array[task] = 0
            continue

        # otherwise get input and output data (only input if non-pseudo sink)
        input_data = list(dag.in_edges(task, data=True))[0][2]['weight']

        if dag.out_degree(task) > 0:
            output_data = list(dag.out_edges(task, data=True))[0][2]['weight']
        else:
            output_data = input_data

        #comp time is the maximum of the two
        comp_array[task] = max(input_data, output_data)

    # build W matrix by transposing and repeating: np.transpose([x]*repeats)
    return np.transpose([comp_array] * num_pes)


def set_streaming_edges_from_streaming_paths(dag, streaming_paths):
    """
    Set the streamability of each edges included in the provided streaming paths

    :param dag: the DAG
    :param streaming_paths: list of edges that are streaming, provided as tuples (src, dst)
    """
    for t in streaming_paths:
        for i in range(len(t) - 1):
            if dag.has_edge(t[i], t[i + 1]):
                # print("Setting: ", t[i], t[i + 1])
                dag[t[i]][t[i + 1]]['stream'] = True
            else:
                print(f"Something is going wrong: edge ({t[i]}, {t[i+1]}) does not exists.")