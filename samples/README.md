## Samples with Synthetic Canonical Task Graph

This folder contains samples using various well-known DAG (chain, Cholesky, Gaussian elimination and FFT).

For each of them, the corresponding script is in charge of creating multiple random graph, schedule them
using S-Sched (and spatial partitioning), and compare the results against the non-stream schedules.
The Scheduling Length Ratio and the Streaming Scheduling Length Ratio are computed as well.

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



### Work in Progress
User-friendly support for switching between two streaming scheduling heuristics.

Currently this must be done by hand by setting the ` create_new_blocks` to either True or False (Relaxed heuristc)
as argument to the call to the function `test_scheduling_heuristics`

