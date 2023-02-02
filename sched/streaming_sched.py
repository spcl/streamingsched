# S-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Streaming Scheduling
    
    This class is in charge of:
    - scheduling a canonical DAG (possibly already partitioned in streaming blocks) according to various heuristics 
        (currently list-based and gang-schedule)
    - analyze the DAG and compute its streaming intervals

    The input DAG must be a single source/single sink graph:
    - the source is always considered a pseudo source (it does not have to be scheduled, but it is convenient to have it)
    - the sink can be a pseudo-sink

    The rest of the DAG is a canonical DAG, that is:
    - it is composed  by elwise/downsampler/upsampler computational nodes 
    - (with the exception of pseudo nodes) all input/output edges incident to a node must have the same volume
    - the DAG may have an arbitrary number of buffer nodes. A buffer node will read the data from its input edge, and *once* all input elements have
been stored, they are output.

'''
import networkx as nx
import numpy as np
from collections import deque, namedtuple
import logging
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass
import math
from fractions import Fraction
from collections import defaultdict
from typing import Set
# A schedule event represent the scheduling of a task, over a Proc. Element, with certain start and
# end time

logger = logging.getLogger('mysched')

# create console handler and set level to debug
if not logger.hasHandlers():
    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[MYSCHED] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


@dataclass
class ScheduleEvent():
    '''

    '''
    task: int
    pe: int
    start_t: int  # Clock cycle granularity for start and end time
    end_t: int
    f_t: int = 0
    api: float = 0  # Actual Production Interval


class StreamingScheduler(object):
    '''
        Streaming Scheduler    
    '''
    def __init__(self, dag: nx.DiGraph, num_pes: int, base_latency: int = 1, buffer_nodes: Set[int] = {}):
        """
        Builds a streaming scheduler

        :param dag: a directed, single source, single sink graph
        :param num_pes: the number of processing elements
        :param base_latency: the base latency needed to compute over a single input datum (by default 1)
            Note: currently it has been not checked for values different than one
        :param buffer_nodes: list of nodes that are buffer nodes and must be treated accordingly
        """

        self.dag: nx.DiGraph = dag
        self.num_pes = num_pes
        self.base_latency = base_latency
        self.num_tasks = len(dag.nodes())
        self.pes_list = list(range(0, self.num_pes))
        self.pes_schedule = dict()
        self.tasks_schedule = dict()
        self.buffer_nodes = buffer_nodes
        # Whether or not we analyzed the DAG for the streaming intervals
        # Note: if this is True it is not guaranteed that the graph has been not changed
        self.streaming_interval_analysis_done = False

        # Find source and sink nodes
        # If the sink node is pseudo, it has a specific attribute ('pseudo') set to True

        # Get exit node
        exit_node = [node for node in self.dag.nodes if self.dag.out_degree(node) == 0]

        # If needed add a pseudo exit node
        assert len(exit_node) == 1, f"Expected a single terminal node, found {len(exit_node)}"
        self.exit_node = exit_node[0]
        self._has_pseudo_exit_node = self._is_pseudo_exit_node(self.exit_node)

        # Get source node
        source_node = [node for node in self.dag.nodes if self.dag.in_degree(node) == 0]
        assert len(source_node) == 1
        self.source_node = source_node[0]

        self.production_rate = dict()

        # TODO: add more check for correctness: for example that all nodes have input edges

        # Currently, we support DAGs that consume/produce the same amount of data from/to all input/output edge
        # This does not apply to source/sink node

        for n in dag.nodes():

            if n == self.source_node or (n == self.exit_node and self._has_pseudo_exit_node):
                # Note: a failure here may indicate that nodes id are not consecutive
                self.production_rate[n] = 1
                continue

            input_data = -1
            for _, _, data in dag.in_edges(n, data=True):
                if input_data == -1:
                    input_data = data['weight']
                else:
                    assert data['weight'] == input_data, f"Node {n} has input edges with different volume"

            output_data = -1
            for _, _, data in dag.out_edges(n, data=True):
                if output_data == -1:
                    output_data = data['weight']
                else:
                    if n not in self.buffer_nodes:
                        assert data[
                            'weight'] == output_data, f"Node {n} has output edges with different volume ({data['weight']} - {output_data})"

            # save the node production rate
            self.production_rate[n] = Fraction(int(output_data), int(input_data))

    def set_logging_level(self, level):
        logger.setLevel(level)

    def set_logging_debug(self):
        logger.setLevel(logging.DEBUG)

    def _is_pseudo_exit_node(self, node):
        """
        Returns wether the node is a pseudo exit node

        :param node: node
        """
        return (node == self.exit_node and 'pseudo' in self.dag.nodes()[node] and self.dag.nodes()[node]['pseudo'])

    def _check_buffer_nodes(self):
        # all the outgoing edge from a buffer node must be set to non-streaming
        for n in self.buffer_nodes:
            for _, _, data in self.dag.out_edges(n, data=True):
                assert 'stream' not in data or data['stream'] == False, f"Buffer node {n} has streaming output edge"

    def _compute_execution_time_isolation(self, task: int):
        """
        Computes the execution time of a task in isolation (no backpressure, w/o considering streaming intervals).
        Since we are in an homogeneous setting, this will be equal for all the PEs

        :param task: task id
        :return: execution time in unit times (e.g., clock cycles)
        """

        # TODO: deal with pseudo source/sink nodes
        if task == self.source_node or self._is_pseudo_exit_node(task) or task in self.buffer_nodes:
            return 0
        else:
            inp_edges = self.dag.in_edges(task, data=True)

            # take the amount of input and output data (for the moment being is the same for all
            # incident inp or outp edges)

            input_data = list(inp_edges)[0][2]['weight']

            comp_time = input_data - 1 + self.base_latency

            if self.dag.out_degree(task) > 0:
                out_edges = self.dag.out_edges(task, data=True)
                output_data = list(out_edges)[0][2]['weight']
                comp_time = max(comp_time, (output_data - 1) + self.base_latency)

            return comp_time

    def ranku(self):
        """
        Computes the upward rank of the tasks in the given dag.
        Nodes (tasks) are annotated with their rank.

        NOTE: Differently than HEFT we don't consider the communication cost (this is a static upward rank).
        The average execution time of task is computed by considering it in isolation.

        :return: annotates graph nodes with their ranku

        Credits: https://github.com/mackncheesiest/heft

        """

        #  Uses a basic BFS approach to traverse upwards through the graph assigning ranku along the way

        # Differently than HEFT, we don't consider communication costs so we don't compute
        # average weights for edge (eq. 3 in [1])

        # The ranku considers the execution time of a task on a PE (homogeneosu)

        # set ranku of exit node: it is equal to its computation time
        nx.set_node_attributes(self.dag, {self.exit_node: self._compute_execution_time_isolation(self.exit_node)},
                               "ranku")

        # Start assigning ranku to all remaining nodes
        queue = deque(self.dag.predecessors(self.exit_node))
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
            while _node_can_be_processed(self.dag, node) is not True:
                try:
                    node2 = queue.pop()
                except IndexError:
                    raise RuntimeError(
                        f"Node {node} cannot be processed, and there are no other nodes in the queue to process instead!"
                    )
                queue.appendleft(node)
                node = node2

            max_successor_ranku = -1
            for succnode in self.dag.successors(node):
                logger.debug(f"\tLooking at successor node: {succnode}")
                logger.debug(f"\t its ranku is {self.dag.nodes()[succnode]['ranku']}")

                # Compute max_successor_ranku = max(c_ij + ranku(j)). Here c_ij is zero
                val = self.dag.nodes()[succnode]['ranku']

                if val > max_successor_ranku:
                    max_successor_ranku = val
            assert max_successor_ranku >= 0, f"Expected maximum successor ranku to be greater or equal to 0 but was {max_successor_ranku}"

            exec_time = self._compute_execution_time_isolation(node)
            # update ranku = w_i + max_successor_ranku
            logger.debug(f"\tAssigning rank to {node}: {exec_time} + {max_successor_ranku}")
            nx.set_node_attributes(self.dag, {node: exec_time + max_successor_ranku}, "ranku")

            queue.extendleft([prednode for prednode in self.dag.predecessors(node) if prednode not in queue])

        for node in self.dag.nodes():
            logger.debug(f"Node: {node}, Rank U: {self.dag.nodes()[node]['ranku']}")

    def rankd(self):
        '''
        Computes the downward rank of the tasks in the given dag.
        Nodes (tasks) are annotated with their rank.

        NOTE: Differently than HEFT we don't consider the communication cost (this is a static downward rank).
        The average execution time of task is computed by considering it in isolation.

        :return: annotates graph nodes with their rankd

        '''

        #   Uses a basic BFS approach to traverse upwards through the graph assigning ranku along the way

        # set ranku of entry node: it is equal to 0
        nx.set_node_attributes(self.dag, {self.source_node: 0}, "rankd")

        # Start assigning ranku to all remaining nodes
        for node in nx.algorithms.topological_sort(self.dag):
            max_pred_rankd = 0
            for prednode in self.dag.predecessors(node):
                if 'rankd' not in self.dag.nodes()[prednode]:
                    import pdb
                    pdb.set_trace()
                logger.debug(f"\tLooking at predecessor node: {prednode}")
                logger.debug(f"\t its rankd is {self.dag.nodes()[prednode]['rankd']}")

                # Compute max_pred_rankd = max(c_ij + rankd(j)). Here c_ij is zero
                val = self.dag.nodes()[prednode]['rankd']
                # if not self._is_edge_streaming(prednode, node):
                #     val += self.dag.edges()[prednode, node]['weight']
                max_pred_rankd = max(max_pred_rankd, val)
            assert max_pred_rankd >= 0, f"Expected maximum predecessor rankd to be greater or equal to 0 but was {max_pred_rankd}"

            average_exec_time = self._compute_average_execution_time_isolation(node)

            # update rankd = w_i + max_successor_rankd
            logger.debug(f"\tAssigning rank to {node}: {average_exec_time} + {max_pred_rankd}")
            nx.set_node_attributes(self.dag, {node: average_exec_time + max_pred_rankd}, "rankd")

        for node in self.dag.nodes():
            logger.debug(f"Node: {node}, Rank D: {self.dag.nodes()[node]['rankd']}")

    def makespan(self):
        '''
        :return: Returns the schedule length
        '''

        # Assuming that time starts from zero, we have to find the maximum finishing time
        makespan = 0
        for k, job in self.tasks_schedule.items():
            makespan = max(makespan, job.end_t)
        return makespan

    def _is_edge_streaming(self, src, dst):
        '''
        Returns whether a given edge is streaming
        :return:
        '''
        return "stream" in self.dag.edges()[src, dst] and self.dag.edges()[src, dst]['stream']

    def schedule_dag(self, streaming_blocks=None):
        """
        Computes the schedule of the Task DAG over a set of PEs.
        Streaming communications may impose tasks to start before the predecessor has finished

        NOTE: by default is uses the streaming intervals info, that must be previously computed
        Otherwise, it can try to understand backpressure (outdated)

        :param: streaming_blocks: list of streaming blocks (list of list). It can be used to prioritize
            nodes instead of using ranku. In each streaming block, nodes are ordered according
            to DAG dependencies
        TODO: define a better ranking procedure to take into account streaming communications
        :return:
        """

        tasks_schedule = {}  # Computed schedule
        pes_schedule = {}  # Computed schedule for each PE (a list of task ScheduleEvent)

        # init
        for i in range(self.num_pes):
            pes_schedule[i] = []

        # check that buffer nodes have non-streaming output
        self._check_buffer_nodes()

        if not streaming_blocks:
            # Nodes are prioritized according to their upward rank

            self.ranku()
            # Sort according to ranku
            sorted_nodes = sorted(self.dag.nodes(), key=lambda node: self.dag.nodes()[node]['ranku'], reverse=True)
        else:

            # Uses streaming blocks to prioritize nodes
            sorted_nodes = []
            from_node_to_streaming_block = dict()
            # sort node according to streaming blocks and their topo order
            for sb in streaming_blocks:

                for task in nx.algorithms.topological_sort(self.dag.subgraph(sb)):
                    sorted_nodes.append(task)

                for task in sb:
                    from_node_to_streaming_block[task] = sb

        logger.debug(f"Nodes sorted according ranku: {sorted_nodes}")

        if sorted_nodes[0] != self.source_node:
            logger.debug(
                "Root node was not the first node in the sorted list. Must be a zero-cost and zero-weight placeholder node. Rearranging it so it is scheduled first\n"
            )
            idx = sorted_nodes.index(self.source_node)
            sorted_nodes[idx], sorted_nodes[0] = sorted_nodes[0], sorted_nodes[idx]

        for node in sorted_nodes:
            logger.debug(f"*************** Scheduling task: {node} ***************")
            unschedulable = ScheduleEvent(node, -1, np.inf, np.inf, np.inf, np.inf)
            min_schedule = unschedulable
            min_EDP = np.inf

            # Compute the list of possible schedule across all PEs, ordered by EFT
            # Currently, if we are using streaming blocks, two nodes in the same streaming block
            # can not be allocated to the same PE (they can stream each other -- we enforce strict coscheduling)
            ordered_schedules = self._computes_ordered_schedules_of_task(
                node,
                tasks_schedule,
                pes_schedule,
                False,
                other_nodes_in_streaming_block=None if streaming_blocks is None else from_node_to_streaming_block[node])

            if len(ordered_schedules) == 0 or ordered_schedules[0] == unschedulable:
                raise RuntimeError(f"Unfeasible scheduling: don't know how to schedule {node}")

            min_schedule = ordered_schedules[0]

            # Update schedules
            tasks_schedule[node] = min_schedule

            pes_schedule[min_schedule.pe].append(min_schedule)
            pes_schedule[min_schedule.pe] = sorted(pes_schedule[min_schedule.pe],
                                                   key=lambda schedule_event:
                                                   (schedule_event.end_t, schedule_event.start_t))

        # Run validity checks
        assert self.validate_schedules(pes_schedule, tasks_schedule, False) == True

        self.pes_schedule = pes_schedule
        self.tasks_schedule = tasks_schedule

        return pes_schedule, tasks_schedule

    def gang_schedule(self, streaming_blocks, analyze=False, reorder_streaming_block=False):
        """
        Compute the schedule by gang_scheduling tasks in streaming blocks:
        - all tasks in a streaming block are executed concurrently
        - only when the last task finishes, we can move to schedule the next streaming block
        :param: streaming_blocks: list of streaming blocks (list of list). 
            In each streaming block, nodes must be  ordered according
            to DAG dependencies. Every streaming block must have at most PEs elements
        :param: _analyze_ whether to analyze or not the DAG and compute the streaming intervals of edges
            (must be used after the DAG has changed)

        NOTE: this must be invoked after that the streaming intervals (considering edge type) have been computed. This
        is done automatically, but no guarantee are provided if the DAG has been modified in the meanwhile

        :return: pes_schedule and tasks_schedule: two dictionaries. The former contains for each PE the list of tasks (and respective
            starting time) assigned to it. The latter contains for each task, its scheduling information (e.g., the PE assigned to it)
        """

        if not self.streaming_interval_analysis_done or analyze:
            self.streaming_interval_analysis()
            self.streaming_interval_analysis_done = True

        tasks_schedule = {}  # Computed schedule
        pes_schedule = defaultdict(list)  # Computed schedule for each PE (a list of task ScheduleEvent)

        block_starting_time = 0  # the time at which the first task of the block can start

        # check that buffer nodes have non-streaming output
        self._check_buffer_nodes()

        # reorder_streaming_block = True

        if reorder_streaming_block:
            # reorder streaming block such that prev node are already scheduled
            already_scheduled_nodes = set()
            blocks = deque()
            # print(streaming_blocks)
            for i in range(len(streaming_blocks)):
                blocks.append(streaming_blocks.pop(0))

            while (len(blocks) > 0):
                sb = blocks.popleft()

                # check that all nodes predecessors are in the same sb or already scheduled
                ok = True
                for node in sb:
                    for pred in self.dag.predecessors(node):
                        if not (pred in sb or pred in already_scheduled_nodes):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    already_scheduled_nodes.update(sb)
                    streaming_blocks.append(sb)
                else:
                    blocks.append(sb)
        # print("Streaming blocks: ", streaming_blocks)

        for sb in streaming_blocks:

            # Pay attention to not use more PE than available
            current_pe = 0  # TODO: investigate what changes if we continue on this.
            # It seems to not have impact, but it should have (when you have indep. components)
            max_end_time = 0  # maximum end time of a task in the current streaming block

            # we need to traverse the sb in topo order
            for task in nx.algorithms.topological_sort(self.dag.subgraph(sb)):

                # TODO: force it to use only the given number of PEs?

                assert task in self.buffer_nodes or task == self.source_node or self._is_pseudo_exit_node(
                    task) or current_pe < self.num_pes

                if self._is_pseudo_exit_node(task) and current_pe >= self.num_pes:
                    pe = self.num_pes - 1  # schedule on the last (this because we already incremented)
                elif task in self.buffer_nodes:
                    # schedule it on the same PE of the latest predecessors: in this way it will not uselessly occupy
                    # a PE preventing another task to be scheduled there (e.g., the buffer node start at time 16,
                    # we allocate it on PE i, then no one else can use that PE even if it is free from time 0 to 16. the
                    # other task will be scheduled after time 16 )
                    max_finishing_time = -1
                    max_pred = -1
                    for pred in self.dag.predecessors(task):
                        if tasks_schedule[pred].end_t > max_finishing_time:
                            max_finishing_time = tasks_schedule[pred].end_t
                            max_pred = pred
                    pe = tasks_schedule[pred].pe
                else:
                    pe = current_pe

                # schedule the task on the PE
                sched = self._compute_eft(tasks_schedule, pes_schedule, task, pe, min_starting_time=block_starting_time)
                # sched = self._compute_eft(tasks_schedule, pes_schedule, task, pe) # TODO: this can produce more compact schedule but may produce stalls -- needs to be investigated
                # Update schedules
                tasks_schedule[task] = sched
                pes_schedule[pe].append(sched)
                pes_schedule[pe] = sorted(pes_schedule[pe],
                                          key=lambda schedule_event: (schedule_event.end_t, schedule_event.start_t))

                max_end_time = max(max_end_time, sched.end_t)

                if task != self.source_node and not self._is_pseudo_exit_node(task) and task not in self.buffer_nodes:
                    # pseudo nodes do not use resources
                    current_pe += 1
                    # current_pe %= self.num_pes

            # The next block can start only when the previous finishes
            block_starting_time = max_end_time

        assert self.validate_schedules(pes_schedule, tasks_schedule, False) == True

        self.pes_schedule = pes_schedule
        self.tasks_schedule = tasks_schedule

        return pes_schedule, tasks_schedule

    def validate_schedules(self, pes_schedule, tasks_schedule, coschedule_streaming_nodes):
        '''
        Correcteness checks to validate the schedule
        :return:
        '''
        for pe in range(len(pes_schedule)):
            for job in range(len(pes_schedule[pe]) - 1):
                first_job = pes_schedule[pe][job]
                second_job = pes_schedule[pe][job + 1]
                assert first_job.end_t <= second_job.start_t, \
                    f"Jobs on a particular PE must finish before the next can begin, but job {first_job.task} " \
                    f"on PE {first_job.pe} ends at {first_job.end_t} and its successor {second_job.task} " \
                    f"starts at {second_job.start_t}"

        # all the streaming communication must satisfy the constraint that the receiver is running when the
        # consumer produces the first result

        warning_to_be_fixed = False
        for node in self.dag.nodes():
            for prednode in list(self.dag.predecessors(node)):
                predjob = tasks_schedule[prednode]
                currjob = tasks_schedule[node]
                if "stream" in self.dag.edges()[prednode, node] and self.dag.edges()[prednode, node]['stream']:

                    if coschedule_streaming_nodes:
                        # Note: technical, it should be ok if predjob.f_t > currjob.start_t
                        # But, since we are trying to enforce that in the schedule, we check it
                        # if predjob.f_t != currjob.start_t:
                        #     import pdb
                        #     pdb.set_trace()
                        assert predjob.f_t == currjob.start_t, \
                            f"Streaming communication between task {node} and {prednode}. The producer must " \
                            f"produce the first data only when the reader is ready to receive ({predjob.f_t} must be " \
                            f"equal than {currjob.start_t})."

                    # This may be due to the case where there are more streaming children and they don't start
                    # at the same time. We try to realign them, but maybe it didn't work well.

                    if predjob.end_t > currjob.end_t and currjob.task != self.exit_node:
                        # from utils.visualize import visualize_dag
                        # visualize_dag(self.dag)
                        # print(f"Warning, problems in the Streaming communication between task {prednode} and {node}. The producer must finish earlier."\
                        #       f"(producer finishes at {predjob.end_t}, while consumer finishes "f"at {currjob.end_t}).")
                        warning_to_be_fixed = True
                        if predjob.end_t > currjob.end_t:
                            import pdb
                            pdb.set_trace()
                        assert predjob.end_t <= currjob.end_t , \
                        f"Streaming communication between task {node} and {prednode}. The producer must " \
                        f"end before the consumer (producer finishes at {predjob.end_t}, while consumer finishes "\
                        f"at {currjob.end_t})."

                else:
                    assert predjob.end_t <= currjob.start_t, \
                        f"Non-Streaming precedence among task {predjob.task} and {currjob.task} is not satisfied." \
                        f"The producer finishes at time {predjob.end_t}, while the consumer starts at time {currjob.start_t}"
        if warning_to_be_fixed:
            print(f"Warning, problems in the Streaming communication between task . The producer must finish earlier")
        return True

    def _has_streaming_predecessor(self, task) -> bool:
        '''
        :param task:
        :return: True if the task has a streaming predecessor
        '''
        inp_edges = self.dag.in_edges(task, data=True)
        for edge in inp_edges:
            if 'stream' in edge[2] and edge[2]['stream']:
                return True
        return False

    def _compute_average_execution_in_schedule(self, task, pe, tasks_schedule) -> int:
        """
        Computes averaged execution time of a task in the given schedule
        We have to go over the inputs and, considering the Actual Initiation Interval, we compute
        the execution time

        :param task: task
        :param pe: the PE
        :param tasks_schedule: current tasks schedule
        :rtype: execution time considering the current schedule
        """

        #
        total_comp_time = 0
        numb = 0

        # Get all input edges if any, and compute execution time
        inp_edges = self.dag.in_edges(task, data=True)

        if task == self.source_node or self._is_pseudo_exit_node(task) or task in self.buffer_nodes:
            # Pseudo nodes and buffer nodes do not need to be scheduled
            return 0
        else:

            max_comp = 0

            # This still work under the assumption of same input volume on all the edges (Canonical DAG).
            # But we have still to look at all the input edges to check if any of them is streaming (streaming interval of
            # non-streaming edges is set to 1)
            # TODO clean this out, it seems that is doing useless job (under the assumption that the input volume is the same
            # we could look only at one of the streaming input edges, not to all)
            AII = 1

            for i, edge in enumerate(self.dag.in_edges(task, data=True)):

                # Compute Actual Init. Interval: if this edge represents a streaming communication,
                # look at the predecessor
                AII = 1
                AII = max(AII, edge[2]['streaming_interval'])
                # print("Streaming interval input to task ", task, ": ", AII)

                input_data = edge[2]['weight']

                max_comp = max(max_comp, AII * (input_data - 1))

            # If this node is an upsampler, we should look at the output data volume

            if self.dag.out_degree(task) > 0:
                out_edges = self.dag.out_edges(task, data=True)
                output_data = list(out_edges)[0][2]['weight']

                # Look at the streaming interval of the output edge, and consider also that
                # This could be for example the case of 0 --> 1 -->2, where 0 is the pseudo root
                # and 0->1 is non-streaming, and 2 is an upsampler where 1-> is streaming.
                # Then 1 is backpressured

                # Note: this is necessary because non-streaming edges have streaming interval 1
                # TODO: if this is no longer true, we can update this part of the code and just look at the
                # input

                output_streaming_interval = list(out_edges)[0][2]['streaming_interval']
                max_comp = max(max_comp, output_streaming_interval * (output_data - 1))

                # print("Ouput Computation time for task ", task, " on PE ", pe, ": ", max_comp, "AII: ", AII)
                if output_data > input_data and output_streaming_interval * (output_data - 1) < AII * (input_data - 1):

                    # This is a very specific corner case that I don't know how to handle in a generic way:
                    # Let's assume that you have something like 0->1->2, with all streams and 2 is an upsampler
                    # Then 2 will takes some time to finish after that it receives the last element and this
                    # depends from the upsampling ratio.
                    # If there is another node afterward then this may be already captured by the
                    # streaming interval

                    # add the upsampling ratio
                    # TODO: maybe here we should add the output streaming interval (R-1)*output_str_interval
                    # print("Corner case...")
                    max_comp += math.ceil((output_data / input_data - 1) * output_streaming_interval)
        # if task == 6:
        #     print("Computation time for task ", task, " on PE ", pe, ": ", max_comp, "AII: ", AII)
        return max_comp + self.base_latency

    def _computes_ordered_schedules_of_task(self,
                                            task,
                                            task_schedules,
                                            pes_schedule,
                                            skip_pes=None,
                                            other_nodes_in_streaming_block=None) -> list:
        """
        Computes all the possible schedules of the given task on all available PEs.
        Returns it ordered by EFT (Expected Finishing Time)

        :param task: the task to schedule
        :param task_schedules: the task schedules built so far
        :param pes_schedule: the PEs schedule
        :param skip_pes: list of PEs that must be skipped
        :return:  returns a list of schedules for the task on the PEs, in non-decreasing order of EFT
        """

        schedules = []
        # We are in the homogeneous case: compute the running time for a single PE
        ctime = self._compute_average_execution_in_schedule(task, 0, task_schedules)
        for pe in range(self.num_pes):
            schedules.append(
                self._compute_eft(task_schedules,
                                  pes_schedule,
                                  task,
                                  pe,
                                  other_nodes_in_streaming_block=other_nodes_in_streaming_block,
                                  computation_time=ctime))

        schedules.sort(key=lambda sched: sched.end_t)

        logger.debug(f"Ordered schedules for task {task} : {schedules}")
        return schedules

    def _compute_eft(self,
                     tasks_schedule,
                     pes_schedule,
                     node,
                     pe: int,
                     min_starting_time=0,
                     other_nodes_in_streaming_block=None,
                     computation_time=None):
        """
        Computes the EFT of a particular node (a task) if it were scheduled on a particular PE
        It does this by first looking at all predecessor tasks of a particular node and determining the earliest time a
        task would be ready for execution (ready_time, EST), also by taking into account streaming communications
        It then looks at the list of tasks scheduled on this particular PE and determines the
        earliest time (after ready_time, AST) a given node can be inserted into this PE's queue

        :param: min_starting_time: the minimum time at which the task can start
        """

        est_time = min_starting_time
        logger.debug(f"Computing EST/EFT for node {node} on PE {pe}")
        unschedulable = ScheduleEvent(node, -1, np.inf, np.inf, np.inf, np.inf)

        # Compute the Earliest Starting Time for this task, by looking at its the predecessors

        # If we are using streaming blocks, we don't want to schedule the task in an already taken PE
        # They could be connected by a streaming path
        # (A more refined logic, would have looked if this is really the case)

        # First: compute earliest starting time by looking at predecessors
        for prednode in list(self.dag.predecessors(node)):
            if prednode not in tasks_schedule:
                import pdb
                pdb.set_trace()
            predjob = tasks_schedule[prednode]
            assert predjob != None, f"Predecessor nodes must be scheduled before their children, but node {node} has an unscheduled predecessor of {prednode}"
            logger.debug(f"\tLooking at predecessor node {prednode} with job {predjob} to determine ready time")

            # is the predecessor connected using a stream?
            if self._is_edge_streaming(prednode, node) and node not in self.buffer_nodes:
                # The predecessor is connected with a stream
                if predjob.pe == pe:
                    ready_time_t = predjob.end_t  # (or predjob.start_t + pe latency)
                else:
                    # otherwise, the minimum starting time is given by the time at which
                    # the predecessor produces the first data
                    ready_time_t = predjob.f_t  # (or predjob.start_t + pe latency)

                    #TODO: is not clear how to compute this
                    # ready_time_t = predjob.start_t + self.base_latency
            else:
                # If the incoming edge is non streaming or this node is a buffer node
                # the EST for this node on the considered PE is
                # the finish time of the predecessor
                ready_time_t = predjob.end_t

            if ready_time_t > est_time:
                est_time = ready_time_t

        logger.debug(f"\tReady time determined to be {est_time}")

        if est_time == -1:
            # can not be scheduled at all in this processor
            logger.debug(f"Node {node} can not be scheduled on {pe}")
            min_schedule = unschedulable
        else:

            # TODO: optimize: all PEs are the same, we can compute this average execution time
            # only once
            if computation_time is None:
                computation_time = self._compute_average_execution_in_schedule(node, pe, tasks_schedule)

            job_list = pes_schedule[pe]

            insertion_slot = True  # wether to use or not to use insertion slots TODO: deal better with this

            # Look for an insertion slot in this PE
            for idx in range(len(job_list)):
                prev_job = job_list[idx]
                if insertion_slot and idx == 0:
                    if (prev_job.start_t - computation_time) - est_time > 0:
                        logger.debug(f"Found an insertion slot before the first job {prev_job} on PE {pe}")
                        job_start = est_time
                        min_schedule = ScheduleEvent(node, pe, job_start, math.ceil(job_start + computation_time))
                        break
                if idx == len(job_list) - 1:
                    # end of the job list
                    job_start = max(est_time, prev_job.end_t)
                    min_schedule = ScheduleEvent(node, pe, job_start, math.ceil(job_start + computation_time))
                    break
                next_job = job_list[idx + 1]

                # Start of next job - computation time == latest we can start in this window
                # Max(est_time, previous job's end) == earliest we can start in this window
                # If there's space in there, schedule in it
                logger.debug(
                    f"\tLooking to fit a job of length {computation_time} into a slot of size {next_job.start_t - max(est_time, prev_job.end_t)}"
                )
                if insertion_slot and (next_job.start_t - computation_time) - max(est_time, prev_job.end_t) >= 0:
                    job_start = max(est_time, prev_job.end_t)
                    logger.debug(
                        f"\tInsertion is feasible. Inserting job with start time {job_start} and end time {job_start + computation_time} into the time slot [{prev_job.end_t}, {next_job.start_t}]"
                    )
                    min_schedule = ScheduleEvent(node, pe, job_start, math.ceil(job_start + computation_time))
                    break
            else:
                # For-else loop: the else executes if the for loop exits without break-ing, which in this case means the number of jobs on this PE are 0
                min_schedule = ScheduleEvent(node, pe, est_time, math.ceil(est_time + computation_time))

            logger.debug(f"\tFor node {node} on PE {pe}, the EFT is {min_schedule}")

            # TODO: understand wether we want to have it with ceiling or not
            if node == self.source_node:
                API = 0
            else:
                API = int(computation_time - self.base_latency + 1)
                if self.dag.out_degree(node) > 0:  # not the sink node

                    output_data = list(self.dag.out_edges(node, data=True))[0][2]['weight']

                    # The actual Production Interval, depends on how frequently we produce data at steady=state
                    API = API / output_data
                else:
                    # consider the input volume then (API is not really important for sink nodes)
                    input_data = list(self.dag.in_edges(node, data=True))[0][2]['weight']
                    API = API / input_data

            min_schedule.api = API
            # TODO: Compute F(T) properly
            if node == self.source_node:
                min_schedule.f_t = 0
            elif node == self.exit_node or node in self.buffer_nodes:
                min_schedule.f_t = int(min_schedule.end_t)
            else:
                input_data = list(self.dag.in_edges(node, data=True))[0][2]['weight']
                if output_data < input_data:
                    # downsampler

                    # in this case we should collect as many data as required and only after we output
                    # consider the actual II (the max)
                    # TODO: the stored AII should correspond to the streaming intervals (if they are used)
                    # If we have streaming intervals >1 also for sources, we can just pick up one of them
                    # Similarly, we can use this info to compute the f_t by looking at the streaming interval of the ouput edges
                    AII = 1

                    for _, _, data in self.dag.in_edges(node, data=True):
                        AII = max(AII, data['streaming_interval'])

                    min_schedule.f_t = math.ceil(min_schedule.start_t + AII * math.ceil(input_data / output_data - 1) +
                                                 self.base_latency)

                else:
                    min_schedule.f_t = int(min_schedule.start_t + self.base_latency)

        return min_schedule

    def get_streaming_blocks(self):
        """
            Returns the streaming blocks of the DAG, as a list of lists  
        """

        topo_order = list(nx.topological_sort(self.dag))

        streaming_blocks = dict()  # node -> streaming component (set)
        streaming_blocks[self.source_node] = {self.source_node}

        for node in topo_order:
            is_streaming = False
            node_streaming_component = {node}
            # get all input edges
            for src, dst, data in self.dag.in_edges(node, data=True):
                if 'stream' in data and data['stream'] == True:
                    is_streaming = True
                    # add the streaming component of the source
                    node_streaming_component |= streaming_blocks[src]

            # update all the streaming components of the predecessors
            for neigh in node_streaming_component:
                if neigh != node:
                    streaming_blocks[neigh] |= node_streaming_component

            streaming_blocks[node] = node_streaming_component

        # get all the streaming components

        streaming_components = list()
        for node in topo_order:
            if streaming_blocks[node] not in streaming_components:
                streaming_components.append(streaming_blocks[node])

        # TODO: if I have two independent nodes, they can be in the same streaming block !!!
        # It is better to use the streaming components returned by the partitioning rather than this one
        # However, we may need to use this anyway
        # A possible idea to do this would be the following:
        # - we traverse the graph by looking at sources:
        #   - if the source has no predecessor in the current streaming component add it
        #   - otherwise add it if all the predecessors that are in the same components are connected with streaming edges
        #   (the others can be also non-streaming as they have been already in another components)
        # - we remove the source and continue

        # Streaming blocks must be ordered in such a way we ensure that a streaming block appears only when all
        # the precedent nodes have been already scheduled

        # TODO: maybe this should stay in a while loop to continue until we fix the ordering

        for i in range(len(streaming_components)):

            component = streaming_components[i]

            # check that all the predecessors of the node in this components are in a block before the current one
            ok = True
            for node in component:
                predecessors = self.dag.predecessors(node)
                for pred in predecessors:
                    if pred in component:
                        continue
                    for j in range(i):
                        if pred in streaming_components[j]:
                            break
                    else:
                        ok = False
                        # print("Not found predecessor ", pred, " of node ", node)
                    if not ok:
                        break
                if not ok:
                    break
            # find where is the predecessor and switch the two components
            if not ok:
                for j in range(i, len(streaming_components)):
                    if pred in streaming_components[j]:
                        # switch
                        streaming_components.insert(j + 1, component)
                        streaming_components.remove(component)
                        break

        return streaming_components

    def streaming_interval_analysis(self,
                                    assume_every_edge_is_streaming=False,
                                    remove_edges=None,
                                    do_no_stream_from_reducers=False):
        """
        Streaming interval analysis

        :param assume_every_edge_is_streaming:  defaults to False
        :param remove_edges: remove the indicated edges from the analysis, defaults to None
        :param do_no_stream_from_reducers: assumes that we do not stream from reducers, defaults to False
        :type do_no_stream_from_reducers: _type_, optional
        """
        '''
        Theorem on the max in WCC
        '''

        # for n in self.dag.nodes():

        #     if n == self.source_node or n == self.exit_node:
        #         self.dag.nodes[n]['ratio'] = 1
        #     else:
        #         in_edges = list(self.dag.in_edges(n, data=True))
        #         input_data = in_edges[0][2]['weight']
        #         out_edges = list(self.dag.out_edges(n, data=True))
        #         output_data = out_edges[0][2]['weight']
        #         self.dag.nodes[n]['ratio'] = Fraction(input_data, output_data)

        ### Set all streaming intervals to 1
        for u, v, data in self.dag.edges(data=True):
            data['streaming_interval'] = 1

        ### Detect streaming blocks
        if assume_every_edge_is_streaming:
            # everything is in the component, except pseudo root and pseudo sink (if any)
            nodes = list(self.dag.nodes)
            nodes.remove(self.source_node)
            if self._has_pseudo_exit_node:
                nodes.remove(self.exit_node)
            streaming_blocks = [nodes]
        else:
            streaming_blocks = self.get_streaming_blocks()

        dag_copy = self.dag.copy()

        ### remove additional edges
        ## TODO this is a very provisional trick
        if remove_edges:
            for src, dst in remove_edges:
                # print("removing ", src, dst)
                dag_copy.remove_edge(src, dst)

        # remove all outgoing edge from a buffer node
        for bn in self.buffer_nodes:
            to_remove = []
            for src, dst in dag_copy.out_edges(bn):
                to_remove.append((src, dst))

            dag_copy.remove_edges_from(to_remove)

        for block in streaming_blocks:
            # print("Component: ", block)

            if len(block) == 1:
                continue

            # Loop over all the WCC
            # TODO: deal with buffer node if any
            # TODO: and full reducer as well

            # remove edge to full reducers
            if do_no_stream_from_reducers:
                for node in block:
                    if dag_copy.out_degree(node) > 0:
                        out_edges = list(dag_copy.out_edges(node, data=True))
                        output_data = out_edges[0][2]['weight']
                        if output_data == 1:
                            # print(node, " is a full reducer")
                            # in_edges = list(dag_copy.in_edges(node, data=True))
                            dag_copy.remove_edges_from(out_edges)
            # from utils.visualize import visualize_dag
            # visualize_dag(dag_copy)

            for subg_nodes in nx.algorithms.weakly_connected_components(dag_copy.subgraph(block)):
                # print("Looking at ", subg_nodes)
                subg = self.dag.subgraph(subg_nodes)
                max_in_volume = -1
                node_with_max_inp = -1
                exit_nodes = [node for node in subg.nodes if subg.out_degree(node) == 0]
                source_nodes = [node for node in subg.nodes if subg.in_degree(node) == 0]

                # find the node with maximum input volume (by looking at the original DAG)
                for node in subg.nodes():
                    if node == self.source_node:
                        input_data = list(self.dag.out_edges(node, data=True))[0][2]['weight']
                    else:
                        in_edges = list(self.dag.in_edges(node, data=True))
                        input_data = in_edges[0][2]['weight']

                        # if this is a sink of this block (but not a buffer node), take a look also at the output volume
                        if node in exit_nodes and node != self.exit_node and node not in self.buffer_nodes:
                            input_data = max(input_data, list(self.dag.out_edges(node, data=True))[0][2]['weight'])

                    if input_data > max_in_volume:
                        max_in_volume = input_data
                        node_with_max_inp = node
                sub_topo_order = list(nx.topological_sort(subg))

                # Then we traverse the WCC and we update the streaming intervals (not the input of the sources or the output of exit nodes)

                # TODO TMP: also skip full reducers
                for node in sub_topo_order:
                    if node in exit_nodes:
                        continue

                    out_edges = list(subg.out_edges(node, data=True))
                    output_data = out_edges[0][2]['weight']

                    for _, _, data in subg.out_edges(node, data=True):
                        data['streaming_interval'] = Fraction(max_in_volume, output_data)

    def get_streaming_depth(self):
        '''
        Computes the streaming depth of the DAG (lower bound to parallel execution with infinite PEs and all streaming edges.

        We define:
        - the level of a node in the graph as L(v) = max(R(v), 1) + max{L(u} s.t. u,v in E(G)
        - the level of a (sub)graph is the maximum level of one of its nodes
        - streaming depth as max_{v}(max_{u\in \WCC{v}} K+(u)) + L(G[WCC{v}])
        '''

        # TODO: deal with buffer nodes (remove outgoing edges)

        # Compute the level of each node, by looking at all wcc seperately
        # TODO: deal with pseudo source

        dag = self.dag.copy()

        levels = [0] * dag.number_of_nodes()

        for n in nx.topological_sort(dag):
            if n == self.source_node:
                continue  # Root level is 0
            else:
                # TODO: what is the right definition of the level?
                max_pred_level = 0
                for pred in dag.predecessors(n):
                    max_pred_level = max(max_pred_level, levels[pred])

                levels[n] = max_pred_level + max(self.production_rate[n], 1)
                # print("Level for node: ", n, ": ", levels[n])
        # Now, look at all WCCs

        # remove the pseudo source node
        dag.remove_node(0)

        # remove the pseudo sink node if any
        if self._has_pseudo_exit_node:
            dag.remove_node(self.exit_node)

        max_k_out_wcc = -1
        wcc_level = 0

        # remove all outgoing edge from buffer nodes
        for bn in self.buffer_nodes:
            to_remove = []
            for src, dst in dag.out_edges(bn):
                to_remove.append((src, dst))

            dag.remove_edges_from(to_remove)

        for subg_nodes in nx.algorithms.weakly_connected_components(dag):
            print(subg_nodes)
            subg = self.dag.subgraph(subg_nodes)
            max_out_volume = -1
            exit_nodes = [node for node in subg.nodes if subg.out_degree(node) == 0]
            source_nodes = [node for node in subg.nodes if subg.in_degree(node) == 0]

            # find the node with maximum input volume (by looking at the original DAG)
            for node in subg.nodes():

                input_data = -1
                if node in source_nodes:
                    # if this is a source of this WCC, we should look also at the input volume

                    if node == self.source_node:
                        input_data = list(self.dag.out_edges(node, data=True))[0][2]['weight']
                    else:
                        in_edges = list(self.dag.in_edges(node, data=True))
                        input_data = in_edges[0][2]['weight']

                if node in exit_nodes:

                    # this is an exit node: take a look at the original DAG if that has some
                    # outgoing edge, otherwise consider the input data
                    if node == self.exit_node:
                        output_data = list(self.dag.in_edges(node, data=True))[0][2]['weight']
                    else:
                        output_data = list(self.dag.out_edges(node, data=True))[0][2]['weight']
                else:
                    # just look inside the WCC
                    output_data = list(subg.out_edges(node, data=True))[0][2]['weight']

                # keep track of the max
                max_out_volume = max(max_out_volume, max(output_data, input_data))
            print("Max out volume: ", max_out_volume)
            # compute the level of this WCC
            level = 0
            for node in exit_nodes:
                if not self._is_pseudo_exit_node(node):
                    level = max(level, levels[node])
                else:
                    # take the max of its predecesssors
                    for pred in subg.predecessors(node):
                        level = max(level, levels[pred])
                print("Level of: ", node, ": ", level)
            print("Max level: ", level, levels)
            # TODO: should we add here the level as well?
            if max_out_volume + level > max_k_out_wcc + wcc_level:
                max_k_out_wcc = max_out_volume
                wcc_level = level

        return max_k_out_wcc + wcc_level

    def get_streaming_depth_no_buffer_nodes(self):
        '''
        Computes the streaming depth of the DAG (lower bound to parallel execution with infinite PEs and all streaming edges.

        In this version we don't consider the presence of buffer nodes. We should anyway loop over the various WCC, after removing 
        the pseudo source graph 

        We define:
        - the level of a node in the graph as L(v) = max(R(v), 1) + max{L(u} s.t. u,v in E(G)
        - the level of a (sub)graph is the maximum level of one of its nodes
        - streaming depth as max_{v}(max_{u\in \WCC{v}} K+(u)) + L(G[WCC{v}])
        '''

        # TODO: deal with buffer nodes (remove outgoing edges)

        # Compute the level of each node, by looking at all wcc seperately
        # TODO: deal with pseudo source

        dag = self.dag.copy()

        levels = [0] * dag.number_of_nodes()

        for n in nx.topological_sort(dag):
            if n == self.source_node:
                continue  # Root level is 0
            else:
                # TODO: what is the right definition of the level?
                max_pred_level = 0
                for pred in dag.predecessors(n):
                    max_pred_level = max(max_pred_level, levels[pred])

                levels[n] = max_pred_level + max(self.production_rate[n], 1)

                # or just the previous + 1
                # levels[n] = max_pred_level + 1
                # print("Level for node: ", n, ": ", levels[n])
        # Now, look at all WCCs

        # remove the pseudo source node
        dag.remove_node(0)

        # remove the pseudo sink node if any
        if self._has_pseudo_exit_node:
            dag.remove_node(self.exit_node)

        max_k_out_wcc = -1
        wcc_level = 0

        for subg_nodes in nx.algorithms.weakly_connected_components(dag):

            subg = self.dag.subgraph(subg_nodes)
            max_out_volume = -1
            exit_nodes = [node for node in subg.nodes if subg.out_degree(node) == 0]
            source_nodes = [node for node in subg.nodes if subg.in_degree(node) == 0]

            # find the node with maximum input volume (by looking at the original DAG)
            for node in subg.nodes():

                input_data = -1
                if node in source_nodes:
                    # if this is a source of this WCC, we should look also at the input volume

                    if node == self.source_node:
                        input_data = list(self.dag.out_edges(node, data=True))[0][2]['weight']
                    else:
                        in_edges = list(self.dag.in_edges(node, data=True))
                        input_data = in_edges[0][2]['weight']

                if node in exit_nodes:

                    # this is an exit node: take a look at the original DAG if that has some
                    # outgoing edge, otherwise consider the input data
                    if node == self.exit_node:
                        output_data = list(self.dag.in_edges(node, data=True))[0][2]['weight']
                    else:
                        output_data = list(self.dag.out_edges(node, data=True))[0][2]['weight']
                else:
                    # just look inside the WCC
                    output_data = list(subg.out_edges(node, data=True))[0][2]['weight']

                # keep track of the max
                max_out_volume = max(max_out_volume, max(output_data, input_data))

            # print("Max out volume: ", max_out_volume)

            # compute the level of this WCC
            level = 0
            for node in exit_nodes:
                if not self._is_pseudo_exit_node(node):
                    level = max(level, levels[node])
                else:
                    # take the max of its predecesssors
                    for pred in subg.predecessors(node):
                        level = max(level, levels[pred])
                # print("Level of: ", node, ": ", level)
            # print("Max level: ", level, levels)
            # TODO: should we add here the level as well?
            if max_out_volume + level > max_k_out_wcc + wcc_level:
                max_k_out_wcc = max_out_volume
                wcc_level = level
        # print("Max level: ", wcc_level)
        return max_k_out_wcc + wcc_level

    def _get_input_edge_id(self, edge):
        '''
        Returns the id of the input edge
        :param edge:
        :return:
        '''

        for i, e in enumerate(self.dag.in_edges(edge[1])):
            if e == edge:
                return i

        raise RuntimeError(f"Input edge {edge} not found")
