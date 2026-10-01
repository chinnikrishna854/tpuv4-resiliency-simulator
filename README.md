# tpuv4-resiliency-simulator
This project models one cloud/distributed-systems problem from Zu et al.,
“Resiliency at Scale: Managing Google’s TPUv4 Machine Learning Supercomputer”
(NSDI 2024): large ML jobs are gang scheduled and therefore need a large set
of healthy resources at the same time. Static interconnects require a
contiguous region; a reconfigurable interconnect can combine healthy capacity
from multiple cubes.
Requirements
- Ubuntu or another normal laptop environment
- Python 3.9+
- matplotlib
Run

python3 -m venv .venv

source .venv/bin/activate

python3 -m pip install matplotlib

python3 tpuv4_simulator.py --outdir results

The command writes three CSV files and three PNG graphs to results/.
Experiments
1. Varies gang-job size from 128 to 4,096 TPUs.
2. Varies pod size while keeping the request at 50% of total capacity.
3. Adds correlated whole-cube failures, an additional scenario not explicitly
   evaluated by the paper’s reported availability curve.
The simulator uses Monte Carlo trials with a fixed seed for reproducibility.
Each TPU independently has a 0.1% failure probability and a 1% occupancy
probability unless an experiment changes the parameter. These are illustrative
assumptions, not measurements of Google’s fleet.
Interpretation
The static model is intentionally strict: a job succeeds only if it finds the
required number of consecutive cubes with every TPU healthy and unoccupied.
The reconfigurable model succeeds when total healthy, free TPU capacity is
large enough, abstracting away the TPUv4 OCS and resilient ICI routing.
Therefore, the simulator demonstrates the availability benefit of flexibility,
but it does not model physical topology, route congestion, communication
patterns, reconfiguration latency, or heterogeneous hardware.
AI-use disclosure


AI assistance was used for brainstorming the simulation abstraction,
generating initial code, and editing explanations. The author
reviewed the paper, checked the assumptions and outputs, and remains
responsible for the final code, results, interpretation, and submission.
