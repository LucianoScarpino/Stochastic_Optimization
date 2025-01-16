import gymnasium as gym
import numpy as np
from typing import List, Tuple


class ShopFloor(gym.Env):
    """
        Dynamic environment for the Job Shop Scheduling Problem
            args:
                prb_instance (InstanceJobShopSetUp): instance of the job shop scheduling problem with setup times
    """
    def __init__(self, prb_instance):
        self.prb_instance = prb_instance
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()
        self.state = None
        
        
    def get_action_space(self) -> gym.spaces:
        """ The action space consists of a tuple of two discrete spaces:
            - Machine selection: n_machines
            - Job selection: n_jobs
        """
        action_space = gym.spaces.Tuple(
            gym.spaces.Discrete(self.prb_instance.n_machines),
            gym.spaces.Discrete(self.prb_instance.n_jobs)
        )
        return action_space

    def get_observation_space(self) -> gym.spaces:
        """ The observation space consists of:
            - Machine state: n_machines x 3 matrix each row is (current job, remaining time, last job (needed for setup time))
            - Job state: n_jobs x 3 matrix each row is (release date, due date, # of remaining operations)
            - Operation eligibility: n_ops vector with 1 if operation is eligible, 0 otherwise
            - Current time: scalar value
        """
        machine_state_low = np.array([-1, 0, -1]) # current job, remaining time, last job (-1 if no job)
        machine_state_high = np.array([self.prb_instance.n_jobs, np.inf, self.prb_instance.n_jobs])
        job_state_low = np.array([0, 0, 0])
        job_state_high = np.array([np.inf, np.inf, np.max(self.prb_instance.n_ops)])

        observation_space = gym.spaces.Tuple(
            gym.spaces.Box(low=machine_state_low, high=machine_state_high, shape=(self.prb_instance.n_machines, 3)),
            gym.spaces.Box(low=job_state_low, high=job_state_high, shape=(self.prb_instance.n_jobs, 3)),
            gym.spaces.MultiBinary(self.prb_instance.n_ops),
            gym.spaces.Discrete(1)
        )

        return observation_space

    def simulate_scheduling(self, schedule):
        obj_func = 0
        return obj_func
    
    def reset(self):
        """ Reset the environment to the initial state """
        # Initialize the machine state
        machine_current_job = np.array(self.prb_instance.machines_initial_state)
        machine_remaining_time = np.zeros(self.prb_instance.n_machines)

        machine_last_job = -1 * np.ones(self.prb_instance.n_machines)
        machine_state = np.stack((machine_current_job, machine_remaining_time, machine_last_job), axis=1)

        # Initialize the job state
        job_release_date = self.prb_instance.df_jobs['release_date'].values
        job_due_date = self.prb_instance.df_jobs['due_date'].values
        job_remaining_ops = np.array(self.prb_instance.n_ops)
        job_state = np.stack((job_release_date, job_due_date, job_remaining_ops), axis=1)

        # Initialize the operation eligibility
        operation_eligibility = np.zeros(len(self.prb_instance.lst_operations))
        idxs = [i for i, op in enumerate(self.prb_instance.lst_operations) if op in self.prb_instance.starting_ops]
        operation_eligibility[idxs] = 1

        # Initialize the current time
        current_time = 0

        # Return the initial state
        info = {}
        state = (machine_state, job_state, operation_eligibility, current_time)
        return state, info

    def step(self, action: List[Tuple[int, int]]) -> Tuple[Tuple, float, bool, dict]:
        """ Execute the action in the environment
            args:
                action List[Tuple[int, int]]: multiple machine selection, job selection
            returns:
                state (tuple): machine state, job state, operation eligibility, current time
                reward (float): reward for the action
                done (bool): True if the episode is finished
                info (dict): additional information
        """
        completed_processing_times = []
        rewards = []
        done = False
        info = {}

        # Get the current state
        machine_state, job_state, operation_eligibility, current_time = self.state

        for machine_id, job_id in action:
            # Get selected machine state (current job, remaining time, last job)
            current_job = machine_state[machine_id][0]
            remaining_time = machine_state[machine_id][1]
            last_job = machine_state[machine_id][2]

            # Get selected job state (release date, due date, remaining operations)
            release_date = job_state[job_id][0]
            due_date = job_state[job_id][1]
            remaining_ops = job_state[job_id][2]

            # Get the operation index for the selected job and machine
            op_idx = self.prb_instance.op_from_j_m[job_id, machine_id]

            # Check if the operation is eligible
            if operation_eligibility[op_idx] == 0:
                rewards.append(-1)
                continue

            # Get the setup time between the last job and the selected job on the selected machine
            setup_time = self.prb_instance.get_setup(machine_id, last_job, job_id)

            # Update the machine state (current job, remaining time, last job) after setup time
            machine_state[machine_id][0] = job_id
            machine_state[machine_id][1] = self.prb_instance.lst_job[job_id][op_idx]['processing_time'] + setup_time
            machine_state[machine_id][2] = job_id

            # Update the job state (release date, due date, remaining operations) after setup time
            job_state[job_id][0] = max(release_date, current_time + setup_time)
            job_state[job_id][1] = due_date
            job_state[job_id][2] -= 1

            # Update the operation eligibility
            operation_eligibility[op_idx] = 0

            # Get the processing time for the operation
            total_operation_time = self.prb_instance.lst_job[job_id][op_idx]['processing_time'] + setup_time
            completed_processing_times.append(total_operation_time)
            

            # Get the reward for the operation TODO: Implement reward function
            rewards.append(0)
            
        # Update the current time CHECK THIS
        current_time = np.max([current_time + t for t in completed_processing_times])

        # Compute the reward
        reward = 0
        if len(rewards) > 0:
            reward = np.mean(rewards)

        # Check if the episode is finished
        if np.all(job_state[:, 2] == 0):
            done = True

        # Update the state
        self.state = (machine_state, job_state, operation_eligibility, current_time)

        return self.state, reward, done, info







    def render(self):
        # Render the environment to the screen using plot operations forest
        self.prb_instance.plot_operations_forest()

    def close(self):
        pass

    def seed(self):
        pass
