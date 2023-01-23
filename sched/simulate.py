# S-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    This file contains an automatic procedure for simulating the executiion
    of a given DAG, under a given schedule.

    NOTE: this is a work in progress

    TODO:
    - organize prints and logging
'''

import simpy
import networkx as nx
import time
from sched.streaming_sched import ScheduleEvent
from utils.graph import is_all_streaming_path
import math
import numpy as np
from typing import List, Dict, Set


class GenericTask(object):
    '''
        Represent a generic Task that can be simulated.
        The task has input and output channels that represent streaming input/output connections.
        Non-streaming connections are modeled by events (the task must wait for the completion of the previous task).
        A task is executed by a specific PE. By default, we assume that a task is able to ingest data 
        on every clock cycle and its internal latency is 1 (II=1, L=1). Then, if there is some backpressure
        effect, or some slow producer, it will read the data at the according rate.
        

        At the end of the computation, the task signals its completion by means of an event.
        Events from other tasks can be used to wait for their completion, or for the PE to be available.

        Assumptions:
        - a task may have multiple inputs, but the input data volume is the same (Restricted DAG)
        - a task may have multiple outputs, but the output data volume is the same (Restricted DAG)


        If the communication mode is synchronous, then each streaming task will send the element and an 
        event to let the sender synchronize (https://stackoverflow.com/questions/36201953/simpy-synchronous-communication-channel)
    '''

    def __init__(
            self,
            env: simpy.Environment,
            task_id,
            pe_id,
            task_end_event: simpy.Event,
            data_to_produce: int = 0,  # to any output edge
            data_to_read: int = 0,  # from any input edge
            input_channels: List[simpy.Store] = None,
            output_channels: List[simpy.Store] = None,
            events_to_wait: List = None,
            synchronous_communications=True,
            buffer_node=False):
        '''
        Constructor of a generic task. The amount of data read and produces will
        determine whether this is an elwise, downsampler or upsampler node.
        :param env: simpy Environment
        :param task_id: unique task id
        :param pe_id: unique id of the PE where the task is executed
        :param task_end_event: event to signal completion of this task
        :param input_channels: input streaming channels
        :param output_channels: output streaming channels
        :param events_to_wait: events to wait for before starting the task
        :param synchronous_communications: whether we want to implement synchronous communications (producer sends when receiver is ready to receive)
        :param buffer_node: indicates whether this is a buffer node or not
        '''
        self.env = env
        self.task_id = task_id
        self.pe_id = pe_id
        self.II = 1  # by default # we want them to read as fast as possible
        self.L = 1  # by default
        self.is_buffer_node = buffer_node

        if not input_channels:  # empty list
            self.input_channels = None
        else:
            self.input_channels = input_channels

        if not output_channels:  #empty list
            self.output_channels = None
        else:
            self.output_channels = output_channels

        self.task_end_event = task_end_event
        self.events_to_wait = events_to_wait
        self.data_to_produce = data_to_produce
        self.data_to_read = data_to_read
        self.synchronous_communication = synchronous_communications

        # determine the type of task according to the ratio between read and produced data. Note, the ratios may be not
        # integer, and therefore we would need to deal with that

        self.ratio = max(1, (data_to_read / data_to_produce))
        self.upsampling_ratio = max(1, data_to_produce / data_to_read)

        # Corner cases
        # Given the current implementation, the task continues until it reads or produces all the data.
        # Data is produced when something is read according to the ratios above. These are integers number and
        # this is a problem if the actual ratio is not an integer number
        # If the task is an upsampler, then it should pretend to read at least
        # as many data it has to produce

        if self.data_to_read < self.data_to_produce and data_to_produce % data_to_read != 0:
            # take into account the discrepancy (upsampling ratio may be > 1)
            self.additional_reads = math.ceil(
                (self.data_to_produce - self.data_to_read * int(self.upsampling_ratio)) / int(self.upsampling_ratio))
        else:
            self.additional_reads = 0

        # Non blocking send/receive: it could happens that the order in which input/output channels are visited
        # may prevent
        self._non_blocking_send = True
        self._non_blocking_receive = True

    def start(self, ):
        '''
        Starts the task
        '''
        self.action = self.env.process(self.run())

        # keep track of some stats
        self.start_time = 0
        self.end_time = 0

    def get_timings(self):
        '''
        Returns start and end time
        :return:
        '''
        return self.start_time, self.end_time

    def run(self):
        '''
        The actual task execution, The task/process activates itself on each clock cycle (or on each unit of time)
        and checks what it has to do: maybe it is time to read something from an input queue
        (if available), or it has to produce something in output, ....

        At the beginning the task may wait to start the actual processing:
        - it waits according to the starting time indicated by the schedule (TODO)
        - it waits until all the non streaming predecessors completed (contained in events_to_wait)
        - it waits until all the task assigned on the same PE and that precede him are completed (events_to_wait)
        - it waits to receive something from the input channels (if any)

        Then it starts to compute, trying to produce and/or consume one input element per unit of time:
        - if it is an elwise, it tries to read and produce on each unit of time
        - if it is a downsampler, it needs to read a certain amount of data before producing the output
        - if it is an upsampler, every time it reads an element, produces a certain amount of data (backpressure)
        :return:
        '''

        # List of items that must be sent out, stored as a pair (elem, time_to_output)
        # Data elements are incrasing integer numbers. (NOTE: these start from zero. Numbering may be not super precise)
        # Note that once the element is produced, it is immediately available to the input
        # of the next streaming task.

        output_queue = []

        # Wait for all the non streaming predecessor tasks
        if self.events_to_wait:
            b = simpy.AllOf(self.env, self.events_to_wait)
            yield b

        # TODO: wait according to the starting time indicated by the schedule

        # Start the processing when the first element arrives (if any)
        ev = None
        if self.input_channels is not None:
            if self.synchronous_communication:

                for in_chan in self.input_channels:
                    i, ev = yield in_chan.get()
                    ev.succeed()
            else:
                # only the data is sent
                for in_chan in self.input_channels:
                    i = yield in_chan.get()
                    #     print(f"Task {self.task_id} received first data  at time {self.env.now}")

            to_read = self.data_to_read + self.additional_reads - 1  # we already read the first one
        else:
            i = 0  # This is a task without streaming inputs
            to_read = self.data_to_read + self.additional_reads

        self.start_time = self.env.now

        #### If this is a buffer node, we can just exit. By assumption, all output edges are non-streaming
        if self.is_buffer_node:
            self.task_end_event.succeed()
            self.end_time = self.env.now
            return

        time_to_output = self.start_time + self.L

        self.actual_upsampling_ratio = self.upsampling_ratio

        if self.ratio == 1:  # elwise or upsampler
            for i in range(int(self.actual_upsampling_ratio)):
                output_queue.append((i, time_to_output))
                time_to_output += 1
            output_counter = int(self.actual_upsampling_ratio)

            # take into account how many data we didn't send because of a float upsampling ratio
            # (next time we will send more if possible)
            self.actual_upsampling_ratio = self.actual_upsampling_ratio - int(
                self.actual_upsampling_ratio) + self.upsampling_ratio
        else:  # downsampler (we need to accumulate a certain amount of data)
            output_counter = 0

        # next time to read: this is equal to II (i.e., 1) if this is an elwise or downsampler
        # otherwise it is given by the upsampling ratio

        # print(self.task_id, " Actual upsampling ratio: ", self.actual_upsampling_ratio, self.upsampling_ratio)
        time_to_next_input = self.start_time + max(self.II, int(self.upsampling_ratio))
        processed = 0

        # Number of accumulated elements from last output: if the task is a downsampler, we need
        # to read a certain amount of elements (according to the downsampling ratio) before producing
        # something in output
        accumulated_elemens = 1  # At the beginning we already read something (from memory if non streaming task)

        read_counter = 0  #used for debug ATM

        # print(
        #     f"** Task {self.task_id} started at time {self.start_time}, {output_queue}, data to produce: {self.data_to_produce}, data to read: {to_read}, time to next input: {time_to_next_input}"
        # )

        # This task is 'alive' until there is some data to produce or data to read
        # print(output_queue)
        first_output = True
        while (processed < self.data_to_produce or to_read > 0):

            current_time = self.env.now

            # Check if there is something to output, and output everything that should be output
            while (len(output_queue) > 0 and current_time >= output_queue[0][1]):

                to_output, expected_time = output_queue.pop(0)

                # if the output is on a streaming channel, push it
                # otherwise nothing happens (the data is "written" to global memory)
                if self.output_channels is not None:

                    if self.synchronous_communication:

                        #     print(f"Task {self.task_id} trying output {to_output} at time {self.env.now}")

                        for out_chan in self.output_channels:
                            sync_comm = self.env.event()
                            yield out_chan.put((to_output, sync_comm))
                            yield sync_comm  # wait for the receiver
                            #     print(
                            #         f"Task {self.task_id} output {to_output} ({expected_time}) at time {self.env.now}")
                            #     print(f"Task {self.task_id} output {to_output} at time {self.env.now}")
                    else:

                        for i, out_chan in enumerate(self.output_channels):
                            # if self.task_id in {1}:
                            #     print(
                            #         f"Task {self.task_id} want to output {to_output} to {i} at time {self.env.now} (queue size: {len(out_chan.items)})...."
                            #     )

                            yield out_chan.put(to_output)

                            ######################################
                            # non-blocking send: we want to send a single output element. We do a first round trying to
                            # send only in channels that are not full, then we do a second one by blocking on the other ones

                            # sent_to = set()

                            # # non blocking round
                            # for i, out_chan in enumerate(self.output_channels):
                            #     # if self.task_id in {2}:
                            #     #     print(f"Task {self.task_id} want to output {to_output} at time {self.env.now}....")
                            #     if len(out_chan.items) < out_chan.capacity:
                            #         yield out_chan.put(to_output)
                            #         sent_to.add(i)
                            # # blocking round
                            # for i, out_chan in enumerate(self.output_channels):
                            #     # if self.task_id in {2}:
                            #     #     print(f"Task {self.task_id} want to output {to_output} at time {self.env.now}....")
                            #     if i not in sent_to:
                            #         yield out_chan.put(to_output)

                            ############################################

                            # if self.task_id in {2}:
                            #     print(f"Task {self.task_id} output {to_output} at time {self.env.now}")

                            # if self.task_id in {2, 7}:
                            #     print(f"Task {self.task_id} output {to_output} to {i} at time {self.env.now} ")
                            # if first_output:
                            #     print(f"Task {self.task_id} output {to_output} to {i} at time {self.env.now} ")
                            #     first_output = False
                processed += 1

            # Check if we have to read and it is time to read
            if current_time >= time_to_next_input and to_read > 0:

                # If the input is streaming, wait for the data (otherwise it is always available)
                # Corner case: additional reads are for the sake of fractional upsampling rates

                if self.input_channels is not None and to_read > self.additional_reads:
                    if self.synchronous_communication:
                        # if self.task_id == 6:
                        # print(f"Task {self.task_id} trying to read ...")
                        for in_chan in self.input_channels:
                            i, ev = yield in_chan.get()
                            ev.succeed()
                        # if self.task_id == 6:
                        # print(f"Task {self.task_id} read at time {self.env.now}")
                    else:
                        start_time = -1

                        for i, in_chan in enumerate(self.input_channels):

                            data = yield in_chan.get()

                        #####################################

                        # non-blocking receive: we want to receive a single input element. We do a first round trying to
                        # receive only from channels that are not empty, then we do a second one by blocking on the other ones

                        # received_from = set()

                        # for i, in_chan in enumerate(self.input_channels):
                        #     # if self.task_id in {2}:
                        #     #     print("currently there are items: ", len(in_chan.items))
                        #     #     print(f"Task {self.task_id} reading from {i} at time {self.env.now}...")
                        #     if len(in_chan.items) > 0:
                        #         data = yield in_chan.get()
                        #         received_from.add(i)
                        #     # if self.task_id in {2}:
                        #     #     print(f"Task {self.task_id} read at time {self.env.now}, {data}")

                        # for i, in_chan in enumerate(self.input_channels):
                        #     if i not in received_from:
                        #         data = yield in_chan.get()

                        ################################################

                current_time = self.env.now  # The time at which we actually read
                accumulated_elemens += 1
                read_counter += 1

                # if self.task_id in {2, 5}:
                #     print(f"[Time: {current_time}] Task {self.task_id}: acc elem: {accumulated_elemens}, {self.ratio}")

                ########################
                # Since we have the ratio that could be a floating point number, we added the isclose clause to get
                # cases where they are mostly equal
                if (accumulated_elemens >= self.ratio
                        or math.isclose(accumulated_elemens, self.ratio)) and output_counter < self.data_to_produce:

                    time_to_output = current_time + self.L

                    # we need to output as many element as the upsampling ratio

                    for i in range(int(self.actual_upsampling_ratio)):

                        # to deal with corner cases (decimal upsampling ratio -- see constructor) we may need to
                        # not generate all the outputs
                        if output_counter < self.data_to_produce:  # corner case
                            output_queue.append((output_counter, time_to_output))

                            # if self.task_id == 7:
                            #     print(
                            #         f"[Time: {current_time}] Task {self.task_id}  output {output_counter}  will be output at {time_to_output}"
                            #     )
                            time_to_output += 1
                            output_counter += 1
                        else:  # corner case
                            # print(self.task_id, " set to_read to zero")
                            to_read = 0  # do not try to read anymore

                    self.actual_upsampling_ratio = self.actual_upsampling_ratio - int(
                        self.actual_upsampling_ratio) + self.upsampling_ratio

                    # Note: the ratio may be non integer, therefore we may not completely reset the accumulated_elems counter
                    accumulated_elemens = accumulated_elemens - self.ratio
                #################################

                time_to_next_input = current_time + max(self.II, int(self.upsampling_ratio))

                to_read -= 1
                # if self.task_id in {6}:
                # print(self.task_id, "----- Time to next input: ", time_to_next_input, to_read)

            if processed < self.data_to_produce or to_read > 0:
                yield self.env.timeout(1)  # wait 1 clock cycle

            # if self.task_id == 1:
            # print(
            #     f"********************** Task {self.task_id}, time {self.env.now},  processed {processed}, to_read: {to_read}"
            # )

        # signal the termination
        self.task_end_event.succeed()
        # print(f"--------------------------- Task {self.task_id} finished at time ", self.env.now)
        self.end_time = self.env.now


class Simulation(object):
    '''
    This class represent the simulation of a DAG according to the given schedule.

    NOTE: tasks must be numbered starting from 0, without gaps
    '''

    def __init__(self,
                 dag: nx.DiGraph,
                 task_schedule: Dict[int, ScheduleEvent],
                 pes_schedule: Dict,
                 channel_capacity: int = 1,
                 root_task: int = 0,
                 synchronous_communication: bool = True,
                 channels_capacities: Dict = None,
                 buffer_nodes: Set = {}):

        # NOte: the dag must have a pseudo sink w/o streaming input edges

        # Number of tasks
        N = len(dag.nodes())

        # sink nodes
        sink_nodes = [node for node in dag.nodes if dag.out_degree(node) == 0]

        # Channel capacity
        chan_cap = channel_capacity

        # if channels_capacities is not None:
        #     print("Using provided buffer space")
        # the remaining channels will have capacity = 1

        # buffer nodes
        self.buffer_nodes = buffer_nodes

        # set of channels organized in a dictionary (task, task) - > Channel
        self.channels = dict()

        # set of tasks
        self.tasks = []

        # Simpy Env
        self.env = simpy.Environment()

        self.root_task = root_task

        # set of events associated with the end of each task
        self.end_events = [simpy.Event(self.env) for i in range(N)]

        # start creating tasks. We have to guarantee:
        # - ordering on PE
        # - DAG constraints
        # - order tasks according time (TODO if needed)
        ordered_tasks = sorted(task_schedule.values(), key=lambda schedule_event: schedule_event.start_t)

        # Check that there are no two tasks scheduled on the same PE are not connected by a streaming path
        # This could happens with list scheduling

        for pe in pes_schedule:
            if len(pes_schedule[pe]) == 0:
                continue

            index = 0
            first_task = pes_schedule[pe][index].task
            # skip non root task
            while first_task == root_task and index < len(pes_schedule[pe]) - 1:
                index += 1
                first_task = pes_schedule[pe][index].task

            for sched in pes_schedule[pe][index + 1:]:
                second_task = sched.task
                if second_task not in buffer_nodes and (not 'pseudo' in dag.nodes()[second_task]
                                                        or dag.nodes()[second_task]['pseudo'] == False):
                    for path in nx.algorithms.all_simple_paths(dag, first_task, second_task):
                        assert is_all_streaming_path(
                            dag, path
                        ) == False, f"Two  nodes connected assigned to the same PE are connected by a streaming path {path}"
                first_task = second_task

        for task in ordered_tasks:
            # skip the pseudo root task
            if task.task == root_task:
                continue

            #skip the pseudo sink task (if any)
            if task == ordered_tasks[-1] and 'pseudo' in dag.nodes()[task.task] and dag.nodes()[task.task]['pseudo']:
                # print("Skip pseudo sink: ", task.task)
                # it should not have streaming input edge

                streaming_inputs = False
                for src, dst, data in dag.in_edges(task.task, data=True):
                    if 'stream' in data and data['stream']:
                        streaming_inputs = True
                if streaming_inputs:
                    print(
                        "Note: for simulating the pseudo sink node should not have streaming input edges. Result may be incorrect."
                    )
                continue

            task_input_channels = []
            task_output_channels = []
            tasks_to_wait = []

            # build the set of streaming input and output channels and tasks that must be completed before this one
            # Note: buffer nodes are not actually scheduled. They will wait until predecessors completed (independently of the
            # type of input edge)
            # NOT APPLIED: Corner case: since we may have weak co-scheduling, two streaming tasks may be allocated in the same
            # PE. In this case, we need to wait for the src, not for the FIFO comm
            for src, dst, data in dag.in_edges(task.task, data=True):
                if task.task not in buffer_nodes and 'stream' in data and data[
                        'stream'] and src != root_task:  # streaming input from root is considered as non streaming
                    # and task_schedule[src].pe != task_schedule[dst].pe:
                    # check if the channel exists, otherwise create it
                    if (src, dst) not in self.channels:

                        if channels_capacities is not None:
                            if (src, dst) in channels_capacities:
                                chan_cap = channels_capacities[(src, dst)]
                            else:
                                chan_cap = 1
                        # print("Channel ", src, " -> ", dst, " has space: ", chan_cap)
                        self.channels[(src, dst)] = simpy.Store(self.env, capacity=chan_cap)
                    # print(f"Task {task.task}, has in input the channel {src} -> {dst}")
                    task_input_channels.append(self.channels[(src, dst)])
                else:
                    # this is a buffer node or has a non-streaming input
                    if src != root_task:
                        # if task.task in buffer_nodes:
                        # print("Buffer node ", task.task, " waits until ")
                        tasks_to_wait.append(self.end_events[src])

            # If we are sending to a buffer node, we have to assume this is not done through streams
            for src, dst, data in dag.out_edges(task.task, data=True):
                if dst not in buffer_nodes and 'stream' in data and data['stream']:
                    # check if the channel exists, otherwise create it
                    if (src, dst) not in self.channels:
                        if channels_capacities is not None:
                            if (src, dst) in channels_capacities:
                                chan_cap = channels_capacities[(src, dst)]
                            else:
                                chan_cap = 1
                        # print("Channel ", src, " -> ", dst, " has space: ", chan_cap)
                        self.channels[(src, dst)] = simpy.Store(self.env, capacity=chan_cap)
                    # print(f"Task {task.task}, has in output the channel {src} -> {dst}")
                    task_output_channels.append(self.channels[(src, dst)])

            # add to the set of tasks that must be waited for, the task previously assigned to this pe (unless they are buffer nodes)
            pe_job_list = pes_schedule[task.pe]
            for i in range(len(pe_job_list)):
                if pe_job_list[i].task == task.task:
                    # get the previous one if any and not the pseudo root
                    if i > 0 and pe_job_list[i - 1].task != root_task and pe_job_list[i - 1].task not in buffer_nodes:
                        tasks_to_wait.append(self.end_events[pe_job_list[i - 1].task])

            # Understand how much data it produces:
            # - whether it is a "source" node (not considering pseudo-root) or not, this task will produce
            #   a certain amount of data. By considering just the produced data, it should be easy (TODO understand)
            #   to deal with task that produce less (or more) data than the one that they read.
            # NOTE: by assumption we are working with Restricted DAGs
            produced_data = 0

            if task.task in sink_nodes:

                # This is a non-pseudo sink node: it is like it produces the same data it reads
                produced_data = list(dag.in_edges(task.task, data=True))[0][2]['weight']
            else:

                for src, dst, data in dag.out_edges(task.task, data=True):
                    # for the moment being, the amount of data is always the same for all outputs
                    assert produced_data == 0 or data['weight'] == produced_data
                    produced_data = data['weight']

            # TODO refactor this part
            # Understand how much data is read

            if dag.in_degree(task.task) == 0:
                import pdb
                pdb.set_trace()
            if 'weight' not in list(dag.in_edges(task.task, data=True))[0][2]:
                print(list(dag.in_edges(task.task, data=True)))
                import pdb
                pdb.set_trace()
            read_data = list(dag.in_edges(task.task, data=True))[0][2]['weight']

            # create the task
            self.tasks.append(
                GenericTask(self.env,
                            task.task,
                            task.pe,
                            data_to_produce=produced_data,
                            data_to_read=read_data,
                            task_end_event=self.end_events[task.task],
                            input_channels=task_input_channels,
                            output_channels=task_output_channels,
                            events_to_wait=tasks_to_wait,
                            synchronous_communications=synchronous_communication,
                            buffer_node=task.task in buffer_nodes))

    def execute(self, print_time=False):

        start_time = time.time()
        # start tasks
        for task in self.tasks:
            task.start()

        self.env.run()

        elapsed_time = (time.time() - start_time) * 1e3

        if print_time:
            print(f"Simulation time (ms) : {elapsed_time:.3f}")

    def get_task_timings(self):
        '''
        :return: a dictionary task_id -> start_time, end_time
        '''

        timings = dict()
        for task in self.tasks:
            timings[task.task_id] = task.get_timings()
        return timings

    def get_makespan(self):
        '''
        Returns the makespan (the difference between the starting and the end of the computation
        :return:
        '''
        start = 0
        end = 0
        timings = dict()
        for task in self.tasks:
            task_start, task_end = task.get_timings()
            if task.task_id != self.root_task and task.task_id not in self.buffer_nodes and task_end == 0:
                print("ERROR: simulation stalls")
                # assert task.task_id != self.root_task and task_end != 0, f"Error on {task.task_id}: {task.get_timings()}"
                return np.infty
            start = min(start, task_start)
            end = max(end, task_end)
        return end - start

    def print_task_timings(self):
        for task in self.tasks:
            start_time, end_time = task.get_timings()
            print(f"Task {task.task_id}, start_t = {start_time}, end_t = {end_time}")
