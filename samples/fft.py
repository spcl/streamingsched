# Streaming-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
      Evaluation over randomly generated FFT Canonical Task Graphs.
'''

import argparse

from sched.streaming_sched import StreamingScheduler
from utils.metrics import *
from utils.graph import *
import random
from dags import fft

from utils.streamability_tests_utils import *
import argparse
import statistics
from utils.testing import *
import time
from multiprocessing import Queue, Process, Manager


def worker(queue, id, num_pes, exhaustive_results, max_work_results, streaming_interval_results, max_work_gang_results,
           streaming_interval_gang_results, all_streams_results, all_non_streams_results, sim_errors, streaming_slrs,
           non_streaming_slrs, simulate):

    while (True):
        i, dag, source_node, sink_node, n_edges = queue.get()

        if i == -1:
            break

        scheduler = StreamingScheduler(dag, num_pes=num_pes)
        # Run test
        exhaustive, max_work, streaming_int, max_work_gang, streaming_int_gang, all_streams, all_non_streams, streaming_slr, non_streaming_slr, sim_error = test_scheduling_heuristics(
            dag, num_pes, source_node, sink_node, n_edges, scheduler, 2, simulate=simulate)
        exhaustive_results[i] = exhaustive
        max_work_results[i] = max_work
        streaming_interval_results[i] = streaming_int
        max_work_gang_results[i] = max_work_gang
        streaming_interval_gang_results[i] = streaming_int_gang
        all_streams_results[i] = all_streams
        all_non_streams_results[i] = all_non_streams
        sim_errors[i] = sim_error
        streaming_slrs[i] = streaming_slr
        non_streaming_slrs[i] = non_streaming_slr


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-N",
        type=int,
        nargs="?",  #
        default=4,
        help="Number of nodes at the first level")
    parser.add_argument("-P", type=int, nargs="?", default=4, help="Number of PEs")
    parser.add_argument("-R", type=int, nargs="?", default=1, help="Number of Runs")
    parser.add_argument("-B", type=float, nargs="?", default=0, help="Beta")
    parser.add_argument('--verbose', help='Print more data', action='store_true', default=False)
    parser.add_argument("-T", type=int, nargs="?", default=4, help="Number of Processes")

    args = parser.parse_args()
    N = args.N
    num_pes = args.P
    R = args.R
    beta = args.B
    verbose = args.verbose
    n_proc = args.T
    W = 2048  # Starting edge weight
    simulate = False  # wether or not validate the results against a discrete time simulation

    # Keep track of all results
    manager = Manager()
    exhaustive_results = manager.list([0] * R)
    max_work_results = manager.list([0] * R)
    streaming_interval_results = manager.list([0] * R)
    max_work_gang_results = manager.list([0] * R)
    streaming_interval_gang_results = manager.list([0] * R)
    all_streams_results = manager.list([0] * R)
    all_non_streams_results = manager.list([0] * R)
    sim_errors = manager.list([0] * R)
    streaming_slrs = manager.list([0] * R)
    non_streaming_slrs = manager.list([0] * R)

    # Multiprocessing: each workers deal with a different DAG
    workers_queues = [Queue()] * n_proc

    # Now start all processes
    processes = []
    for i in range(n_proc):
        processes.append(
            Process(target=worker,
                    args=(workers_queues[i], i, num_pes, exhaustive_results, max_work_results,
                          streaming_interval_results, max_work_gang_results, streaming_interval_gang_results,
                          all_streams_results, all_non_streams_results, sim_errors, streaming_slrs, non_streaming_slrs,
                          simulate)))
        processes[i].start()

    start_time = time.time()

    for r in range(R):
        random.seed(r)  # same seed for reproducibility purposes
        dag, source_node, sink_node = fft.build_fft_dag(N, W, ccr=True)

        scheduler = StreamingScheduler(dag, num_pes=num_pes)
        n_edges = dag.number_of_edges() - 2 * N

        workers_queues[r % n_proc].put((r, dag, source_node, sink_node, n_edges))

    # Stop workers
    for i in range(n_proc):
        workers_queues[i].put((-1, -1, -1, -1, -1))

    for i in range(n_proc):
        workers_queues[i].close()
        workers_queues[i].join_thread()

    for i in range(n_proc):
        processes[i].join()

    skip_exhaustive = exhaustive_results[0] == 0

    # compute statistics
    if skip_exhaustive:
        res_exhaustive = 0
    else:
        res_exhaustive = statistics.geometric_mean(exhaustive_results)

    if max_work_results[0] == 0:
        res_max_work = 0
        res_max_work_gang = 0
    else:
        res_max_work = statistics.geometric_mean(max_work_results)
        res_max_work_gang = statistics.geometric_mean(max_work_gang_results)

    res_streaming_interval = statistics.geometric_mean(streaming_interval_results)
    res_streaming_interval_gang = statistics.geometric_mean(streaming_interval_gang_results)

    res_all_streams = statistics.geometric_mean(all_streams_results)
    res_all_non_streams = statistics.geometric_mean(all_non_streams_results)

    res_streaming_slrs = statistics.median(streaming_slrs)
    res_non_streaming_slrs = statistics.median(non_streaming_slrs)

    if verbose:
        print("Elapsed time (s): ", time.time() - start_time)

        print("Simulation report:")
        print("\t min error: ", min(sim_errors))
        print("\t max error: ", max(sim_errors))
        print("\t med error: ", statistics.median(sim_errors))

        if num_pes >= dag.number_of_nodes() - 2:
            print(
                f"streaming depth speedup: get_type_of_nodes(dag, source_node, sink_node){res_all_streams*num_pes:.2f}")

        # currently we don't show max work since it is the same as streaming interval
        if res_max_work != res_streaming_interval:
            print("Max work produces different results ", res_max_work, res_streaming_interval)
    if not simulate:
        print("## One-line Summary. Medians of: \nStreaming-Sched speedup\tNon-Streaming Speedup\tSSLR\tSLR")
        print(
            f"{res_streaming_interval_gang*num_pes:.2f}\t{res_all_non_streams*num_pes:.2f}\t{res_streaming_slrs:.2f}\t{res_non_streaming_slrs:.2f}"
        )
    else:
        print(
            "## One-line Summary. Medians of: \nStreaming-Sched speedup\tNon-Streaming Speedup\tSSLR\tSLR\tError wrt Simulation"
        )
        print(
            f"{res_streaming_interval_gang*num_pes:.2f}\t{res_all_non_streams*num_pes:.2f}\t{res_streaming_slrs:.2f}\t{res_non_streaming_slrs:.2f}\t{statistics.median(sim_errors):.4f}"
        )

    # Save results to file

    #organize the data as list of lists
    data = []
    sim_err_csv = []
    for r in range(R):
        str_interval_speedup = round(streaming_interval_results[r] * num_pes, 3)  # not used
        gang_str_interval_speedup = round(streaming_interval_gang_results[r] * num_pes, 3)
        non_stream_speedup = round(all_non_streams_results[r] * num_pes, 3)
        sim_err = round(sim_errors[r], 5)
        streaming_slr = round(streaming_slrs[r], 3)
        non_streaming_slr = round(non_streaming_slrs[r], 3)

        data.append([gang_str_interval_speedup, non_stream_speedup, streaming_slr, non_streaming_slr, sim_err])
        sim_err_csv.append([sim_err])

    results_filename = f'results_fft_N_{N}_P_{num_pes}.csv'
    sim_error_filename = f'sim_error_fft_N_{N}_P_{num_pes}.csv'

    results_header = ["Streaming_Sched_Speedup", "NonStream_Speedup", "StreamingSLR", "NonStreamingSLR", "Sim_Error"]
    sim_error_header = [f"{num_pes}"]

    save_results_to_file(results_filename, results_header, data)
    save_sim_errors_to_file(sim_error_filename, sim_error_header, sim_err_csv)
