## Samples with Synthetic Canonical Task Graph

This folder contains samples using various well-known DAG (chain, Cholesky, Gaussian elimination and FFT).

For each of them, the corresponding script is in charge of creating multiple random graph, schedule them
using Streaming-Sched (and spatial partitioning), and compare the results against the non-streaming schedule.
The Scheduling Length Ratio (SLR) and the Streaming Scheduling Length Ratio (SSLR) are computed as well.

All the scripts can be invoked as follows:

```Bash
$ python samples/<script_name>.py -N 8 -R 10 -P 4 -T 4
```
where:
- $N$ is the parameter controlling the DAG size. For the chain task graph is the number of tasks. For the other dags, please refer to the documentation 
    under the `dags` repo subfolder
- $R$ is the number of synthetic task graph to generate
- $P$ is the number of PEs to use for the schedule
- $T$ (optional) is the number of threads to be used for evaluating the various DAGs. Using multi-threading is suggested for larger/numerous graphs.

The sample runs till completion, and prints on the standard output the median speedup for streaming and non-streaming schedules, SLR and SSLR.
The results for each generated task graph are saved in a `csv` file.

### Buffer space computation and validation

For each sample, the user can compute also the buffer space and validate the computed schedule against a Discrete Event Simulation of the same program, to assess the correctness of buffer space computation for pipelined communications (i.e., the simulation does not deadlock), and the quality of results (the steady-state analysis allows us to compute a realistic makespan).

The validation must be manually enabled by the user, by setting the `simulate` boolean flag to `True` in the sample source code.

### Work in Progress
User-friendly support for switching between two streaming scheduling heuristics.

Currently this must be done by hand by setting the ` create_new_blocks` to either True or False (Relaxed heuristc)
as argument to the call to the function `test_scheduling_heuristics`

