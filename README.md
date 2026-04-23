# Split-Execution of a Heavy Task in a Decentralized System under Non-Uniform Component Distribution and Uniform Event Arrival Rate

## Problem Statement

Consider a set of $N$ devices $D_1, D_2, D_3, \dots, D_N$ forming a mesh network. A heavy computation task is divided into $K$ components $C_1, C_2, \dots, C_K$.

Each component $C_i$ is installed in one of the devices following a non-uniform geometric distribution. In other words, the probability that a device is installed with $C_i$ is not $1/K$, but rather a function of $K$ indicating that different components are distributed with different proportions. For instance, the 1st component has the highest ratio, the second one has a lower ratio, and so on. The chosen distribution satisfies:

$$p_1 > p_2 > p_3 > \dots > p_K$$

and

$$\sum_{i=1}^{K} p_i = 1$$

where $p_i$ represents the relative proportion or load associated with component $C_i$.

Tasks originate randomly near any of the $N$ nodes with a probability of $1/N$. The task arrivals follow a stochastic process. However, one complete execution of a task requires contribution from all $K$ components in sequence. After completing component $C_i$, the task is forwarded through the mesh network to the device hosting $C_{i+1}$.

Because events appear randomly near any node, and due to multi-hop forwarding communication delays and sequential execution requirements, queues will build up at each of the nodes bearing some component $C_i$. Each component-hosting device maintains a service queue for incoming task requests.

## System Parameters (Inputs)

* $N$: Number of devices in the network

* $K$: Number of components per task

* $A$: Length of one side of the deployment area (network area = $A \times A$)

* $R$: Communication radius of each device

* Distribution: A non-uniform distribution defining the relative load or placement probability of each component

* $\lambda$ (Task arrival rate): Average rate at which tasks are generated in the network

* $\mu$ (Service rate): Processing capacity of each component

* Routing model: Multi-hop shortest-path or flooding-based forwarding

* Execution order: Sequential execution of components ($C_1 \rightarrow C_2 \rightarrow \dots \rightarrow C_K$)

## Target Metrics (Outputs)

In this context, the analysis must find the following:

* Average Time ($T$): What is the average time taken by one task to complete execution under this split-execution mode?

* Average Queue Size ($L$): What is the average size of the queue under each of the components?

* Maximum Sustainable Arrival Rate ($\lambda_{max}$): What is the maximum arrival rate the system can handle before becoming unstable?

* Delay Breakdown: What fraction of the total delay is caused by queueing at components versus communication between components?

* Impact of Non-Uniformity: How does the non-uniformity of the distribution affect the overall delay and queue sizes?

## Evaluation & Results

The theoretical queueing and network models are verified using a discrete-event simulation (implemented in Python/Jupyter Notebook/C++) to compare and validate the theoretical queueing results against simulation-based results.

## Installation & Setup

It is recommended to use a Python environment with Python version 3.12 or greater. Set up a virtual environment with the command:

```python -m venv .venv```
and activate it:
```source .venv/bin/activate``` (for Unix/Linux/Mac) 
```./.venv/Scripts/activate``` (for Windows).

The libraries are mentioned in the `requirements.txt` file. You can install them using pip:

```pip install -r requirements.txt```