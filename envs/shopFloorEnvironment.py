import gymnasium as gym
import numpy as np

class ShopFloorEnvironment(gym.Env):
    """ Base class for the shop floor scheduling environment
        args:
            prb_instance: The problem instance (see data_interfaces.py)
    """
    def __init__(self, prb_instance):
        self.prb_instance = prb_instance

    def get_state_space(self) -> gym.spaces.Dict:
        """ Construct the state space of the environment 
            job_state (n_jobs, 2): The state of the jobs [current_operation, status]
            machine_state (n_machines, 2): The state of the machines [job_idx, remaining_time]
            schedule_state (n_jobs, max_ops, 4): The schedule state [start_time, duration, machine_idx, status]
            current_time (1): The current time
        """
        n_jobs = self.prb_instance.n_jobs
        max_ops = max(self.prb_instance.n_ops)
        n_machines = self.prb_instance.n_machines   
        state_space = gym.spaces.Dict({
            'job_state': gym.spaces.Box(
                low=np.tile(np.array([-1, 0], dtype=np.int32), (n_jobs, 1)),  
                high=np.tile(np.array([max_ops-1, 1], dtype=np.int32), (n_jobs, 1)), 
                dtype=np.int32
            ),
            'machine_state': gym.spaces.Box(
                low=np.tile(np.array([-1, 0], dtype=np.int32), (n_machines, 1)),  
                high=np.tile(np.array([n_jobs, 10**6], dtype=np.int32), (n_machines, 1)), 
                dtype=np.float32  
            ),
            'schedule_state': gym.spaces.Box(
                low=np.tile(np.array([-1, -1, -1, -1], dtype=np.int32), (n_jobs, n_machines, 1)),  
                high=np.tile(np.array([10**6, 10**6, n_machines, 2], dtype=np.int32), (n_jobs, n_machines, 1)),  
                dtype=np.float32
            ),
            'current_time': gym.spaces.Discrete(1)  # Current time as a discrete value
        })
        return state_space
    
    def reset(self):
        """ Reset the environment to the initial state """
        
        job_state = np.zeros((self.prb_instance.n_jobs, 2), dtype=np.int32)
        machine_state = np.zeros((self.prb_instance.n_machines, 2), dtype=np.int32)
        machine_state[:, 0] = self.prb_instance.machines_initial_state

        # Initialize the current schedule
        max_operations = max(self.prb_instance.n_ops)
        current_schedule = np.zeros((self.prb_instance.n_jobs, max_operations, 4))
        for job_idx, num_ops in enumerate(self.prb_instance.n_ops):
            current_schedule[job_idx, num_ops:, :] = [-1, -1, -1, -1]

        state = {
            'job_state': job_state,
            'machine_state': machine_state,
            'schedule_state': current_schedule,
            'current_time': 0}
        return state