# S-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Collections of utils for carrying on tests /samples
'''

import networkx as nx
from multiprocessing import Queue, Process, Manager
from utils.metrics import makespan
from multiprocessing import Queue, Process, Manager
import copy
import statistics
import sys
import pdb
import numpy as np
from sched.utils import print_schedule, build_W_matrix_HEFT, set_streaming_edges_from_streaming_paths
from utils.visualize import show_schedule_gantt_chart, visualize_dag
from sched.simulate import Simulation
from utils.graph import build_restricted_nx, save_to_file, read_from_file
from collections import defaultdict
from sched import heft
from sched import eft
from utils.graph import set_all_edges_type
from sched.deadlock_prevention import compute_buffer_space
import time


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def set_streams(dag, source_node, sink_node, stream_values: list, detect_unvalid=False, num_pes=1):
    '''
    Given a DAG and a list of dags, set 

    :param dag:
    :param source_node:
    :param sink_node:
    :param stream_values:
    :param detect_unvalid:
    :param num_pes: meaningful if detect unvalid set to true
    :return:
    '''
    # get edges that do not touch pseudo root
    candidate_edges = []
    for e in dag.edges():
        if e[0] != source_node and e[1] != sink_node:
            candidate_edges.append(e)
    # streaming_components = [1] * dag.number_of_nodes()
    streaming_components = dict()
    for n in dag.nodes():
        streaming_components[n] = {n}
    # print("Len: ", len(candidate_edges))
    # print(candidate_edges)
    # print("Len: ", len(stream_values))
    # task start from 1 till N-1
    for i in range(len(stream_values)):
        source = candidate_edges[i][0]
        sink = candidate_edges[i][1]
        dag[source][sink]['stream'] = stream_values[i]
        if detect_unvalid and stream_values[i]:
            if len(streaming_components[source] | streaming_components[sink]) <= num_pes:

                # streaming_components[source] += streaming_components[sink]
                # streaming_components[sink] += streaming_components[source]

                # TODO find a more efficient way of keeping track of this
                to_add = list(streaming_components[source])
                neighbours = list(streaming_components[source])
                for n in neighbours:
                    streaming_components[n] |= set(streaming_components[sink])
                streaming_components[source] |= set(streaming_components[sink])

                neighbours = list(streaming_components[sink])
                for n in neighbours:
                    streaming_components[n] |= set(to_add)
                streaming_components[sink] |= set(to_add)
            else:
                return False

    return True


def run_exhaustive_search(dag,
                          n_proc,
                          n_edges,
                          source_node,
                          sink_node,
                          StreamingScheduler,
                          num_pes,
                          detect_unvalid=False):
    '''
    Runs a multithreaded exaustive saech
    '''
    dag_copies = []
    for i in range(n_proc):
        dag_copies.append(copy.deepcopy(dag))

    manager = Manager()
    results = manager.dict()
    errors = manager.list([0] * n_proc)

    workers_queue = [Queue()] * n_proc

    # Now start all processes
    processes = []
    for i in range(n_proc):
        processes.append(
            Process(target=worker,
                    args=(workers_queue[i], i, dag_copies, source_node, sink_node, StreamingScheduler, results, errors,
                          detect_unvalid, num_pes)))
        processes[i].start()

    all_stream_values = []
    for i in range(2**(n_edges)):
        bin_string = "{0:b}".format(i).zfill(n_edges)
        lista = [b == '1' for b in bin_string]
        workers_queue[i % n_proc].put((i, lista))

    # finish workers
    for i in range(n_proc):
        workers_queue[i].put((-1, []))

    for i in range(n_proc):
        workers_queue[i].close()
        workers_queue[i].join_thread()

    for i in range(n_proc):
        processes[i].join()

    return results, errors


def test_scheduling_heuristics(dag,
                               num_pes,
                               source_node,
                               pseudo_sink_node,
                               n_edges,
                               scheduler,
                               exaustive_nedge_limit,
                               buffer_nodes={},
                               simulate=False,
                               verbose=False,
                               run=-1,
                               create_new_blocks=True):
    """
    Tests various streaming blocks heuristics, exaustive search

    This is similar to 'test_all_stream_choice_heuristic', but smaller and run also the exhaustive search

    :param dag: the dag
    :param num_pes: number of PEs
    :param source_node:  the pseudo root node
    :param pseudo_sink_node: the pseudo sink node (or -1 if it does not exists)
    :param n_edges: 
    :param scheduler: [description]
    :param exhaustive_nedge_limit: max number of edge to allowed to run the exhaustive search
    :param simulate: whether simulate or not
    :param create_new_blocks: controls the heuristic that will be used
    """

    from sched import spatial_block_partitioning, StreamingScheduler

    print_timings = False
    # Keep track of all results
    set_streams(dag, source_node, pseudo_sink_node, [False for i in range(n_edges)])

    #############################
    # Exhaustive search

    n_proc = 8

    if n_edges <= exaustive_nedge_limit:
        results, errors = run_exhaustive_search(dag, n_proc, n_edges, source_node, pseudo_sink_node, StreamingScheduler,
                                                num_pes)

        # Show the first 10 results
        if verbose:
            print("Top-10 results for exhaustive search: ")

            for k, v in sorted(results.items(), key=lambda item: item[1])[0:10]:
                print(k, v)
        keys, top_exaustive = sorted(results.items(), key=lambda item: item[1])[0]

        if verbose:
            print("Errors", sum(errors))
    else:
        if verbose:
            print("Too many edge to do an exhaustive search")
        top_exaustive = 0

    max_work = 0
    max_work_gang = 0

    ##############################
    # Streaming intervals

    set_all_edges_type(dag, False, exclude_pseudo_nodes=False)
    # set_streams(dag, source_node, pseudo_sink_node, [False for i in range(n_edges)])
    start_t = time.time()
    streaming_paths_str_int, streaming_components_str_int = spatial_block_partitioning.spatial_block_partitioning(
        dag, num_pes, source_node, pseudo_sink_node, buffer_nodes=buffer_nodes, create_new_blocks=create_new_blocks)
    if print_timings:
        print("Spatial block partitioning: ", time.time() - start_t)
    set_streaming_edges_from_streaming_paths(dag, streaming_paths_str_int)
    # print("streaming paths: ", streaming_paths_str_int)
    # print("Str int compo", streaming_components_str_int)
    # save_to_file(dag, '/tmp/gaussian.json')

    scheduler = StreamingScheduler(dag, num_pes=num_pes, buffer_nodes=buffer_nodes)
    start_t = time.time()
    scheduler.streaming_interval_analysis()
    if print_timings:
        print("Interval analsysis: ", time.time() - start_t)

    # for i, (u, v, data) in enumerate(dag.edges(data=True)):
    #     print(f"{u} -> {v} : {data['streaming_interval']}")
    start_t = time.time()
    pes_schedule_str_int, tasks_schedule_str_int = scheduler.schedule_dag(streaming_components_str_int)
    if print_timings:
        print("List scheduling time: ", time.time() - start_t)

    # print("Streaming, off-chip memory accesses: ",
    #   compute_off_chip_memory_accesses(dag, source_node, pseudo_sink_node, buffer_nodes))
    streaming_interval = makespan(tasks_schedule_str_int)

    if verbose:
        print("### Streaming interval stream choice heuristics: ", streaming_interval)

    ## Gang Scheduling
    start_t = time.time()
    pes_schedule_gang_str_int, tasks_schedule_gang_str_int = scheduler.gang_schedule(streaming_components_str_int)
    if print_timings:
        print("Gang scheduling time: ", time.time() - start_t)

    streaming_interval_gang = makespan(tasks_schedule_gang_str_int)

    # Compute streaming depth:
    str_depth = scheduler.get_streaming_depth_no_buffer_nodes()

    ############
    # All streams and all not streams

    # In this case we cant' really evaluate all streaming because both cases (w/ - w/o streaming intervals
    # consider always that you can stream even if data is in DRAM. This is an upper bound to an actual
    # schedule
    set_streams(dag, source_node, pseudo_sink_node, [True for i in range(n_edges)])

    # set back edges going out from buffer nodes to non streaming
    for bn in buffer_nodes:
        for _, _, data in dag.out_edges(bn, data=True):
            data['stream'] = False

    # visualize_dag(dag)
    all_stream_scheduler = StreamingScheduler(dag, num_pes=dag.number_of_nodes(), buffer_nodes=buffer_nodes)
    start_t = time.time()
    all_stream_scheduler.streaming_interval_analysis()
    if print_timings:
        print("all stream analysis: ", time.time() - start_t)
    # scheduler.steady_state_analysis()

    blocks = [dag.nodes()]
    start_t = time.time()
    try:

        pes_schedule, tasks_schedule = all_stream_scheduler.gang_schedule(streaming_blocks=blocks)
        all_streams = makespan(tasks_schedule)
    except Exception as e:
        print(e)
        print("Unable to stream with all streams")
        all_streams = 0
    if verbose:
        print("### All streams : ", all_streams)
    if print_timings:
        print("all stream scheduling: ", time.time() - start_t)

    ### Non stream scheduling: use HEFT

    set_all_edges_type(dag, False, exclude_pseudo_nodes=False)
    start_t = time.time()
    W = build_W_matrix_HEFT(dag, source_node, pseudo_sink_node, buffer_nodes, 1)
    heft_pes_schedule, heft_tasks_schedule = eft.schedule_dag(dag, W, num_pes)
    depth = dag.nodes[0]['ranku']
    all_non_streams = makespan(heft_tasks_schedule)

    streaming_slr = float(streaming_interval_gang / str_depth)
    non_streaming_slr = float(all_non_streams / depth)

    if print_timings:
        print("Heft scheduling time: ", time.time() - start_t)

    scheduler = StreamingScheduler(dag, num_pes=1, buffer_nodes=buffer_nodes)
    seq_time = 0
    for node in nx.topological_sort(dag):
        if node == source_node or node == pseudo_sink_node:
            continue
        seq_time += scheduler._compute_execution_time_isolation(node)

    ########################################################
    # Simulation
    ########################################################
    if simulate:
        # this is computed considering one of the streaming heuristics as reference
        # currently we also consider negative error (if negative, simulation time is lower than schedule time)
        # TODO: pay attention to insertion slot due to s-heft

        # set all edge to non stream then set them according to the computed streaming paths
        set_all_edges_type(dag, False, exclude_pseudo_nodes=False)
        set_streaming_edges_from_streaming_paths(dag, streaming_paths_str_int)
        scheduler = StreamingScheduler(dag, num_pes=num_pes, buffer_nodes=buffer_nodes)
        scheduler.streaming_interval_analysis()

        buff_space = 0

        channels_capacities = compute_buffer_space(dag, streaming_components_str_int, tasks_schedule_gang_str_int,
                                                   source_node)

        # SET BACK SOME STREAM TO NON STREAMING

        # change streaming interval, by reomving edges with weight > buffer_space
        # This should reduce a bit the issues that we have with outliers, but does not solve completely

        iterations = 5  # Having more than one iteration makes sense only if we recompute the streaming blocks

        for it in range(iterations):
            remove_edges = []
            for src, dst, data in dag.edges(data=True):
                if 'stream' in data and data['stream']:
                    if data['weight'] > 1 and data['weight'] - 1 <= channels_capacities[src, dst]:
                        # print("Removing ", (src, dst))
                        dag.edges[src, dst]['stream'] = False
                        remove_edges.append((src, dst))
                    elif data['weight'] == 1:
                        # print("Removing reducer edge", (src, dst))
                        dag.edges[src, dst]['stream'] = False
                        remove_edges.append((src, dst))

            # set_streaming_edges_from_streaming_paths(dag, streaming_paths_str_int)
            # Now, recompute stream and buffer space
            if len(remove_edges) == 0:
                break
            scheduler = StreamingScheduler(dag, num_pes=num_pes, buffer_nodes=buffer_nodes)
            scheduler.streaming_interval_analysis()

            # If we want to iterate, we need to recompute the streaming blocks. But this is not so easy to do
            streaming_components_str_int = scheduler.get_streaming_blocks()
            pes_schedule_gang_str_int, tasks_schedule_gang_str_int = scheduler.gang_schedule(
                streaming_components_str_int, reorder_streaming_block=False)

            channels_capacities = compute_buffer_space(dag, streaming_components_str_int, tasks_schedule_gang_str_int,
                                                       source_node)

        pes_schedule_gang_str_int, tasks_schedule_gang_str_int = scheduler.gang_schedule(streaming_components_str_int)
        streaming_interval_gang = makespan(tasks_schedule_gang_str_int)

        sim = Simulation(dag,
                         tasks_schedule_gang_str_int,
                         pes_schedule_gang_str_int,
                         buff_space,
                         channels_capacities=channels_capacities,
                         root_task=source_node,
                         synchronous_communication=False,
                         buffer_nodes=buffer_nodes)
        sim.execute(print_time=False)

        simulation_time = sim.get_makespan()

        if simulation_time == np.infty:
            # What to do with this: currently we just skip it for testing
            sim_error = 1
        else:
            sim_error = (simulation_time - streaming_interval_gang) / simulation_time
    else:
        sim_error = 0

    if top_exaustive == 0:
        eff_top_exaustive = 0
    else:
        eff_top_exaustive = (seq_time / top_exaustive) / num_pes
    if max_work == 0:
        eff_max_work = 0
    else:
        eff_max_work = (seq_time / max_work) / num_pes
    eff_streaming_interval = (seq_time / streaming_interval) / num_pes
    if max_work_gang == 0:
        eff_max_work_gang = 0
    else:
        eff_max_work_gang = (seq_time / max_work_gang) / num_pes
    eff_streaming_interval_gang = (seq_time / streaming_interval_gang) / num_pes
    eff_all_streams = (seq_time / all_streams) / num_pes
    eff_all_non_streams = (seq_time / all_non_streams) / num_pes
    return eff_top_exaustive, eff_max_work, eff_streaming_interval, eff_max_work_gang, eff_streaming_interval_gang, eff_all_streams, eff_all_non_streams, streaming_slr, non_streaming_slr, sim_error
