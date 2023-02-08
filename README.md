[![CI Tests](https://github.com/spcl/streamingsched/actions/workflows/python-package.yml/badge.svg)](https://github.com/spcl/streamingsched/actions/workflows/python-package.yml)

# Streaming-Sched

This repository contains Scheduling heuristics for Streaming Task Graphs over DataFlow Architectures.

On Spatial devices (such as CGRAs) the computation can be performed both _spatially_, by taking advantage
of a large number of computing units, and _temporally_, by time-multiplexing resources to perform the computation.
On top of this dicothomy, _pipelining_ can be crucial to fully exploit the device’s spatial parallelism.

Streaming-Sched propose models and heuristics for scheduling a direct acyclic task graph on dataflow devices, specifically dealing with *time vs space* tradeoff, 
and considering  *pipelining* across tasks as first-class citizen desiderata.


## Code - Requirements and Setup



The library is written on Python3.8 and successive.

To install all the required modules it is sufficient to run from the repository folder:

```
pip install -r requirements.txt
```


## Usage

### Canonical DAGs

Streaming-Sched accepts Direct Acyclic Graphs in the form of `Networkx` DiGraph. 
The nodes of the graph represent *tasks* in which the application can be decomposed. 
An edge between two nodes $u$ and $v$ indicates a data dependency between two tasks, and it is annotated with the amount of data being
transferred between $u$ and $v$. An edge can be *streaming* (meaning that sender and receiver can be simultaneously in execution) or *non-streaming* (the receiver will be executed only when the sender completed its execution).

To be schedulable using the proposed heuristic, the DAG must be in a canonical form.
This means that:
- the DAG has a single source and a single sink node (they can be pseudo-node not performing actual work)
- each of its _computational_ node receives $I$ data elements from all the input edges and produces $O=RI$ data elements to its output edge. The constant $R$ indicates the production rate of the node. 
- the time that it will take for a given node to compute, depends on the amount of data being read or produced. It will take one time unit per element being produced or read. A computational node can _stream_ the output element as soon as they are ready (without waiting for the completion of the entire task).
- the DAG may have an arbitrary number of _buffer_ nodes. A buffer node will read the data from its input edge, and *once* all input elements have
been stored, they are output $R$ times.

The following example shows the creation of a chain of four *elementwise* tasks (all the task read and produce the same amout of data)

```Python
import networkx as nx
dag = nx.DiGraph()
# Add nodes 
dag.add_node(0)
dag.add_node(1)
dag.add_node(2)
dag.add_node(3)
dag.add_node(4)
dag.add_node(5 ,pseudo=True) # Sink is a pseudo node

# Add edges, indicating data volume (weight) and whether the edge is streaming or not. By default the edge is assumed to be non-streaming.
dag.add_edge(0, 1, weight=4, stream=False)
dag.add_edge(1, 2, weight=4, stream=False)
dag.add_edge(2, 3, weight=4, stream=False)
dag.add_edge(3, 4, weight=4, stream=False)
dag.add_edge(4, 5, weight=4, stream=False)
```

Nodes may optionally have a `label` attribute, and the DAG can be visualized in DOT format:

```Python
from utils.visualize import visualize_dag
visualize_dag(dag)
```

### Scheduling a DAG with user hints

Once a Canonical DAG has been created, it can be scheduled on $P$ processing elements, using the proposed heuristics.

By default the scheduler will use the streaming information provided by the user and embedded in the DAG. These comprise:
- the spatial block partitioning of the DAG: a partition of the graph in components having at most $P$ tasks that will be co-scheduled;
- wether or not an edge is streaming.

Once the DAG has been constructed, its scheduling can be derived as follows:

```Python
# dag: the DAG to be scheduled
# num_pes: number of Processing Elements to use
# buffer_nodes: set of buffer nodes, e.g. {1,2,3, ...}
# streaming_blocks: list of set of spatial blocks, e.g., [{1, 2, 3}, {4, 5, 6}]
scheduler = StreamingScheduler(dag, num_pes=8, buffer_nodes=buffer_nodes)
pes_schedule, tasks_schedule = scheduler.gang_schedule(streaming_blocks)
```

The scheduling function returns a pair of dictionaries. The former contains for each PE the list of tasks (and respective
starting time) assigned to it. The latter contains for each task, its scheduling information (e.g., the PE assigned to it).

This information can be printed on the standard output or visualized with a Gantt chart

```Python
from sched.utils import print_schedule
from utils.visualize import visualize_dag
print_schedule(pes_schedule, "Tasks") # prints on the standard output
show_schedule_gantt_chart(pes_schedule) # visualizes a Gantt chart
```



### Streaming Scheduling with spatial partitioning

How to partition the DAG into spatial blocks is not straightforward. Streaming-Sched comes with heuristics that do this with the aim of minimizing the makespan.

Once the user has created the DAG, the spatial block partitioning can be invoked as follows:
```Python

from sched.spatial_block_partitioning import spatial_block_partitioning
# Spatial block partitioning: returns the list of streaming edges and the spatial blocks
streaming_paths, spatial_blocks = spatial_block_partitioning(
        dag, num_pes, pseudo_source_node, pseudo_sink_node, buffer_nodes=buffer_nodes)

# Apply those changes to the DAG
from sched.utils import set_streaming_edges_from_streaming_paths
set_streaming_edges_from_streaming_paths(dag, streaming_paths)

# Schedule the DAG
scheduler = StreamingScheduler(dag, num_pes=8, buffer_nodes=buffer_nodes)
pes_schedule, tasks_schedule = scheduler.gang_schedule(streaming_blocks)
```

### Buffer space computation and validation

Despite Streaming-Sched considers direct _acyclic_ graph, deadlocks can still occur in the presence of streaming communications if insufficient buffer space is used.

Therefore we provide to the users an analysis pass to inspect the given task graph and compute schedule and return the buffer space
for each streaming edge.

```Python
from sched.deadlock_prevention import compute_buffer_space
buffers_space = compute_buffer_space(dag, spatial_blocks, tasks_schedule, source_node)
```

The `compute_buffer_space` function returns a dictionary, containing for each streaming edge $(src, dst)$ the corresponding buffer space

Streaming-Sched uses Discrete Event Simulation to assess the correctness of buffer space computation for pipelined communications (the simulation does not deadlock), and the quality of results (the makespan of the computed schedule is close to the simulated one).
The Discrete Event Simulation is implemented in `simpy` and takes into account the task graph, the spatial partitioning and the PE assignment of each task as decided by the scheduling heuristic.

```Python
from sched.simulate import Simulation
sim = Simulation(dag,
                tasks_schedule,
                pes_schedule,
                channels_capacities=buffers_spaces,
                buffer_nodes=buffer_nodes)
sim.execute(print_time=False)
simulated_makespan = sim.get_makespan()
```

The simulation returns the simulated makespan, that can be compared with the one returned by the Streaming-Sched heuristics.


## Samples and tests

This repository provides functions to create DAG with well-known structures, together with samples and unit tests for basic functionalities:

### Synthetic Canonical Task Graphs

The `dags` subfolder, contains functions to create Canonical Task Graphs generated from well-known computations: tasks chain, Fast Fourier Transform, Gaussian Elimination, and Tiled Cholesky Factorization. 
For a given topology, edge weights are randomly generated.

For example, to generate a random chain of tasks:

```Python
from dags import chain
N = 8 # Number of Tasks
W = 128 # Output volume of the first task
dag, source_node, sink_node = chain.build_chain_dag(N=N, W=W, random_nodes=True)
```
Please refer to the documentation in the appropriate module for the specific argument to pass to the creation function.


### Evaluation and validation
The `samples` folder contains samples using various well-known DAG. Please refer to the relative README.

The `tests` directory contains unit tests for validating basic functionalities.


## Development and Contribution
Streaming-Sched is an open-source project. We are happy to accept Pull Requests with your contributions! Please follow the contribution guidelines before submitting a pull request.











