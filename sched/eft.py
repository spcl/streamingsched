# S-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    Implementation of Earliest Finish Time (HEFT) scheduling algorithm, for the homogeneous case.

    This is a critical path list-based scheduling algorithm (based on HEFT).

    Sources:
    [1] "Performance-effective and low-complexity task scheduling for heterogeneous computing" - Topclougu et al, 2002.
    [2] https://en.wikipedia.org/wiki/Heterogeneous_Earliest_Finish_Time
    [3] Practical Multiprocessor Scheduling Algorithms for Efficient Parallel Processing, kasahara 1984

    For inputs it takes a set of tasks, represented as a directed acyclic graph, the number of  Processing Elements,
    the times to execute each task.

    There are no communication costs.

"""

import networkx as nx
import numpy as np
from collections import deque, namedtuple
import logging
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
from dataclasses import dataclass

# ScheduleEvent = namedtuple('ScheduleEvent', 'task proc start_t end_t')

logger = logging.getLogger('heft')

# create console handler and set level to debug
if not logger.hasHandlers():
    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[HEFT] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def set_logging_level(level):
    logger.setLevel(level)


# A schedule event represent the scheduling of a task, over a Proc. Element, with certain start and
# end time
@dataclass
class EFTScheduleEvent():
    '''

    '''
    task: int
    pe: int
    start_t: float
    end_t: float


def schedule_dag(dag: nx.DiGraph, W: np.array, P: int):
    '''
    Computes the schedule of the Task Dag over a set of P PEs
    :param dag:
    :param W: computation cost array of size #Task
    :param P: num processing elements
    :return: a PE and task schedule
    '''

    # TODO: deal with non-single source/sink node

    NT = W.shape[0]  # Number of Tasks
    task_schedules = {}  # Computed schedule
    pes_schedule = {}  #Computed schedule for each PE (a list of task EFTScheduleEvent)

    # init
    for i in range(P):
        pes_schedule[i] = []

    # Get source node
    source_node = [node for node in dag.nodes if dag.in_degree(node) == 0]
    assert len(source_node) == 1
    source_node = source_node[0]

    # Compute upward rank
    ranku(dag, W)

    topo_sort = list(nx.topological_sort(dag))
    # Sort according to ranku
    # Note: if we have buffer nodes, these have running time equal to zero. So it could occur that the sorted nodes do not
    # respect the topological order (e.g., you have two buffer nodes one after the other, they can be switched)
    # We take care of this by already starting from the node in topological order, and relying on the sorting algorithm being stable
    # sorted_nodes = sorted(topo_sort, key=lambda node: dag.nodes()[node]['ranku'], reverse=True)

    # For having the CP/MISF variants: in case two tasks have the same priority, the task having the largest number of
    # immediately successive tasks is assigned the highest priority
    sorted_nodes = sorted(topo_sort, key=lambda node: (dag.nodes()[node]['ranku'], dag.out_degree(node)), reverse=True)

    # TODO: deal with non-single source/sink node
    if sorted_nodes[0] != source_node:
        logger.debug(
            "Root node was not the first node in the sorted list. Must be a zero-cost and zero-weight placeholder node. Rearranging it so it is scheduled first\n"
        )
        idx = sorted_nodes.index(source_node)
        sorted_nodes[idx], sorted_nodes[0] = sorted_nodes[0], sorted_nodes[idx]

    for node in sorted_nodes:
        min_task_schedule = EFTScheduleEvent(node, -1, np.inf, np.inf)
        min_EDP = np.inf

        # Find the PE that minimize the EFT of the task. The PEs are homoheneous but a PE may be occupied
        # in doing something elese
        for pe in range(P):
            task_schedule = compute_eft(dag, task_schedules, pes_schedule, W, node, pe)
            if (task_schedule.end_t < min_task_schedule.end_t):
                min_task_schedule = task_schedule
        # Update schedules
        task_schedules[node] = min_task_schedule
        pes_schedule[min_task_schedule.pe].append(min_task_schedule)
        pes_schedule[min_task_schedule.pe] = sorted(pes_schedule[min_task_schedule.pe],
                                                    key=lambda schedule_event:
                                                    (schedule_event.end_t, schedule_event.start_t))

        # correctness check
        for pe in range(len(pes_schedule)):
            for job in range(len(pes_schedule[pe]) - 1):
                first_job = pes_schedule[pe][job]
                second_job = pes_schedule[pe][job + 1]
                assert first_job.end_t <= second_job.start_t, \
                    f"Jobs on a particular PE must finish before the next can begin, but job {first_job.task} " \
                    f"on PE {first_job.pe} ends at {first_job.end} and its successor {second_job.task} " \
                    f"starts at {second_job.start}"
    return pes_schedule, task_schedules


def compute_eft(dag, task_schedules, pes_schedule, W, node, pe):
    """
    Computes the EFT of a particular node (a task) if it were scheduled on a particular PE
    It does this by first looking at all predecessor tasks of a particular node and determining the earliest time a
    task would be ready for execution (ready_time, EST)
    It then looks at the list of tasks scheduled on this particular PE and determines the
    earliest time (after ready_time, AST) a given node can be inserted into this PE's queue
    :param W: computation cost matrix, a matrix #Task x #PEs. The entry Wij indicates the cost of
        executing task i on PE j
    :param C: communication cost matrix, a matrix #PEs x #PEs. The entry Cij indicates the data transfer rate between
        PEs i and j.
    :param L: communication startup cost L, array of size #PEs, indicating the cost for initiating a communication
        from PE i

    """
    ready_time = 0
    # No communication cost
    communication_cost = 0
    logger.debug(f"Computing EFT for node {node} on PE {pe}")

    # Compute the Earliest Executing starting Time for this task, by looking at its the predecessors
    for prednode in list(dag.predecessors(node)):
        if prednode not in task_schedules:
            import pdb
            pdb.set_trace()
        predjob = task_schedules[prednode]
        assert predjob != None, f"Predecessor nodes must be scheduled before their children, but node {node} has an unscheduled predecessor of {prednode}"
        logger.debug(f"\tLooking at predecessor node {prednode} with job {predjob} to determine ready time")

        ready_time_t = predjob.end_t  # no comm cost

        logger.debug(f"\tNode {prednode} can have its data routed to PE {pe} by time {ready_time_t}")
        if ready_time_t > ready_time:
            ready_time = ready_time_t
    logger.debug(f"\tReady time determined to be {ready_time}")

    computation_time = W[node][0]
    job_list = pes_schedule[pe]
    for idx in range(len(job_list)):
        prev_job = job_list[idx]
        if idx == 0:
            if (prev_job.start_t - computation_time) - ready_time > 0:
                logger.debug(f"Found an insertion slot before the first job {prev_job} on PE {pe}")
                job_start = ready_time
                min_schedule = EFTScheduleEvent(node, pe, job_start, job_start + computation_time)
                break
        if idx == len(job_list) - 1:
            # end of the job list
            job_start = max(ready_time, prev_job.end_t)
            min_schedule = EFTScheduleEvent(node, pe, job_start, job_start + computation_time)
            break
        next_job = job_list[idx + 1]

        # Start of next job - computation time == latest we can start in this window
        # Max(ready_time, previous job's end) == earliest we can start in this window
        # If there's space in there, schedule in it
        logger.debug(
            f"\tLooking to fit a job of length {computation_time} into a slot of size {next_job.start_t - max(ready_time, prev_job.end_t)}"
        )
        if (next_job.start_t - computation_time) - max(ready_time, prev_job.end_t) >= 0:
            job_start = max(ready_time, prev_job.end_t)
            logger.debug(
                f"\tInsertion is feasible. Inserting job with start time {job_start} and end time {job_start + computation_time} into the time slot [{prev_job.end_t}, {next_job.start_t}]"
            )
            min_schedule = EFTScheduleEvent(node, pe, job_start, job_start + computation_time)
            break
    else:
        # For-else loop: the else executes if the for loop exits without break-ing, which in this case means the number of jobs on this PE are 0
        min_schedule = EFTScheduleEvent(node, pe, ready_time, ready_time + computation_time)
    logger.debug(f"\tFor node {node} on PE {pe}, the EFT is {min_schedule}")
    return min_schedule


def ranku(dag: nx.DiGraph, W: np.array):
    '''
    Computes the upward rank of the tasks in the given dag.
    Nodes (tasks) are annotated with their rank.

    Since we are not considering (for the moment being) the communication cost,
    this is a static upward rank
    :param dag: the task Direct Acyclic Graph. Edges (i,j) are annotated with a weight corresponding to the
        amount of data transferred from task i to task j
    :param W: computation cost matrix, a matrix #Task x #PEs. The entry Wij indicates the cost of
        executing task i on PE j
    :param C: communication cost matrix, a matrix #PEs x #PEs. The entry Cij indicates the data transfer rate between
        PEs i and j.
    :param L: communication startup cost L, array of size #PEs, indicating the cost for initiating a communication
        from PE i
    :return:
    '''

    #TODO: deal with non schedulable task on given PE

    #   Uses a basic BFS approach to traverse upwards through the graph assigning ranku along the way

    # Get exit node
    exit_node = [node for node in dag.nodes if dag.out_degree(node) == 0]

    # exit_node = [node for node in dag.nodes() if not any(True for _ in dag.successors(node))]
    # If needed add a pseudo exit node
    assert len(exit_node) == 1, f"Expected a single terminal node, found {len(exit_node)}"
    exit_node = exit_node[0]

    # no communication cost (average of matrix B in [1])
    # This can not be zero, since it is used to divide the communication cost
    avg_communication_cost = 1

    # set ranku of exit node: it is equal to its computation time
    nx.set_node_attributes(dag, {exit_node: W[exit_node][0]}, "ranku")
    queue = deque(dag.predecessors(exit_node))
    while queue:
        node = queue.pop()

        def _node_can_be_processed(dag, node):
            """
            Validates that a node is able to be processed in Rank U calculations. Namely, that all of its successors
            have their Rank U values properly assigned. Otherwise, errors can occur in processing DAGs of the form
            A
            |\
            | B
            |/
            C
            Where C enqueues A and B, A is popped off, and it is unable to be processed because B's Rank U has
            not been computed
            """
            for succnode in dag.successors(node):
                if 'ranku' not in dag.nodes()[succnode]:
                    return False
            return True

        logger.debug(f"Assigning ranku for node: {node}")
        # Get next node to process
        while _node_can_be_processed(dag, node) is not True:
            try:
                node2 = queue.pop()
            except IndexError:
                raise RuntimeError(
                    f"Node {node} cannot be processed, and there are no other nodes in the queue to process instead!")
            queue.appendleft(node)
            node = node2

        max_successor_ranku = -1
        for succnode in dag.successors(node):
            logger.debug(f"\tLooking at successor node: {succnode}")
            logger.debug(f"\tThe edge weight from node {node} to node {succnode} is {0}, "
                         f"and the ranku for node {node} is {dag.nodes()[succnode]['ranku']}")

            # Compute max_successor_ranku = max(c_ij + ranku(j)). Here c_ij is zero
            val = dag.nodes()[succnode]['ranku']
            if val > max_successor_ranku:
                max_successor_ranku = val
        assert max_successor_ranku >= 0, f"Expected maximum successor ranku to be greater or equal to 0 but was {max_successor_ranku}"

        # update ranku = w_i + max_successor_ranku
        logger.debug(f"\tAssigning rank to {node}: {W[node][0]} + {max_successor_ranku}")
        nx.set_node_attributes(dag, {node: W[node][0] + max_successor_ranku}, "ranku")

        queue.extendleft([prednode for prednode in dag.predecessors(node) if prednode not in queue])

    for node in dag.nodes():
        logger.debug(f"Node: {node}, Rank U: {dag.nodes()[node]['ranku']}")


def show_Gantt_chart(pes_schedule, task_labels=None, PE_labels: list = None):
    """
        Given a dictionary of PE-task schedules, displays a Gantt chart generated using Matplotlib
        :param task_labels: mapping node_id->label
        :param PE_labels: labels to use for PE (must be compliant with pe_schedule)
    """

    PEs = list(pes_schedule.keys())

    color_choices = ['red', 'blue', 'green', 'cyan', 'magenta']

    if PE_labels is not None:
        # build color array: one color per Processing Element type
        PEs_colors = []
        used_colors = {}
        color_idx = 0
        for idx, p in enumerate(PE_labels):
            if p not in used_colors:
                used_colors[p] = color_idx
                color_idx = color_idx + 1
            PEs_colors.append(used_colors[p])
    ilen = len(PEs)
    pos = np.arange(0.5, ilen * 0.5 + 0.5, 0.5)
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(111)
    for idx, pe in enumerate(PEs):
        for job in pes_schedule[pe]:
            job_duration = job.end_t - job.start_t
            if job_duration > 0:  # Do not show jobs with duration = 0
                ax.barh((idx * 0.5) + 0.5,
                        job_duration,
                        left=job.start_t,
                        height=0.3,
                        align='center',
                        edgecolor='black',
                        color='white',
                        alpha=0.95)

                label = str(job.task + 1) if task_labels is None else task_labels[job.task]
                label_color = color_choices[((job.task) // 10) %
                                            5] if PE_labels is None else color_choices[PEs_colors[job.pe]]
                ax.text(0.4 * job_duration + job.start_t, (idx * 0.5) + 0.5 - 0.03125,
                        label,
                        color=label_color,
                        fontweight='bold',
                        fontsize=16,
                        alpha=0.75)

    locsy, labelsy = plt.yticks(pos, PE_labels)
    plt.ylabel('PE', fontsize=16)
    plt.xlabel('Time (clock cycles)', fontsize=16)
    plt.setp(labelsy, fontsize=14)
    ax.set_ylim(ymin=-0.1, ymax=ilen * 0.5 + 0.5)
    ax.set_xlim(xmin=-5)
    ax.grid(color='g', linestyle=':', alpha=0.5)

    font = font_manager.FontProperties(size='small')
    plt.savefig('/tmp/gannt.png')
    plt.show()
