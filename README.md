# PolitoStochastic

## Overview
**PolitoStochastic** is a repository for simulating machine failures in job shop scheduling using various priority rules. The simulation generates an animated Gantt chart of the dynamic scheduling environment.

---

## How to Run

### 1. Initialize Virtual Environment
Set up a Python virtual environment and install the required dependencies:

```bash
python3 -m venv <path_to_venv>
source <path_to_venv>/bin/activate  # Activate the virtual environment
pip install -r requirements.txt
```

### 2. Run the Simulation
Execute the following command to run the simulation:

```bash
python3 main.py --priority_rule <rule> --failure_prob <probability> --n <iterations>
```

- **`--priority_rule <rule>`**: The priority rule to apply for scheduling.
- **`--failure_prob <probability>`**: Probability of machine failure during the simulation.
- **`--n <iterations>`**: Number of iterations to simulate.

Example:

```bash
python3 main.py --priority_rule edd --failure_prob 0.1 --n 100
```

---

## Implementation

### Environment
The environment is implemented in `ShopFloorEnvironment` and consists of the following components:

1. **Job State** (`np.ndarray(n_jobs, 2)`): Represents the current status of jobs.
    - `job_state[job_idx]`: `[current_op_idx, is_running]`
        - `current_op_idx`: Index of the current operation (-1 if idle).
        - `is_running`: `0` (idle) or `1` (running).

2. **Machine State** (`np.ndarray(n_machines, 2)`): Represents the current status of machines.
    - `machine_state[machine_idx]`: `[current_job_idx, remaining_time]`
        - `current_job_idx`: Index of the job being processed by the machine.
        - `remaining_time`: Remaining time for the current operation (0 if idle).

3. **Schedule State** (`np.ndarray(n_jobs, n_machines, 4)`): Represents the current schedule.
    - `schedule[job_idx, machine_idx]`: `[start, duration, machine_idx, status]`
        - `start`: Start time of the operation.
        - `duration`: Remaining time for the operation.
        - `machine_idx`: Index of the assigned machine.
        - `status`: Status of the operation:
            - `0`: Waiting to be processed.
            - `1`: Being processed.
            - `2`: Completed.
        - Unscheduled operations are represented as `[-1, -1, -1, -1]`.

4. **Current Time**: The current simulation time.

---

### Simulation
The simulation is implemented in `ShopFloorSimulation`, a subclass of `ShopFloorEnvironment`. It includes methods for:
- Simulating machine failures.
- Scheduling operations based on a provided priority list.

---

### Job Schedulers and Agents
The scheduling logic is implemented in two types of agents:
1. **StaticAgent**: Uses static priority rules.
2. **DynamicAgent**: Adapts scheduling based on dynamic conditions.

Both agents inherit from `JobShopScheduler`, which ensures that all schedules adhere to classic job shop scheduling constraints.

## Data 

The dataset is located in `data/I-5x10-equal-loose-0/`, and the class `InstanceJobShopSetUp` contains all static information related to the data. See `instances/instanceJobShopSetUp.py` for details.

## Structure
```
├── agents
│   ├── __init__.py
│   ├── agents.py
│   └── scheduler.py
├── data
│   └── I-5x10-equal-loose-0
│       ├── exact_sol_details.json
│       ├── initial_setup.csv
│       ├── jobs.csv
│       ├── operations.csv
│       ├── settings.json
│       └── setup.csv
├── data_interfaces
│   ├── __init__.py
│   └── read_file.py
├── envs
│   ├── __init__.py
│   ├── shopFloorEnvironment.py
│   ├── shopFloorGantt.py
│   └── shopFloorSimulation.py
├── instances
│   ├── __init__.py
│   ├── instance.py
│   ├── instanceJobShop.py
│   └── instanceJobShopSetUp.py
├── main.py
```
