import gymnasium as gym
import numpy as np
from typing import List, Tuple

class ShopFloorRL(gym.Env):
    """
        Dynamic environment for the Job Shop Scheduling Problem
            args:
                prb_instance (InstanceJobShopSetUp): instance of the job shop scheduling problem with setup times
                failure_prob (float): probability of machine failure

        TODO:
            - current time should be updated as the next time a machine is available
            - the rewards of each action in the list of actions should be computed when the action is executed
    """
    def __init__(self, prb_instance, failure_prob=0.0):
        self.prb_instance = prb_instance
        self.failure_prob = failure_prob
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()
        self.state = None
    
    def get_action_space(self) -> gym.spaces.Dict:
        """ The action space consists of a dictionary where:
            - 'machines': MultiBinary space indicating which machines are active.
            - 'jobs': MultiBinary space indicating which jobs are selected for processing.
            - 'do_nothing': Discrete space indicating if no action is taken.
        """
        action_space = gym.spaces.Dict({
            'machines': gym.spaces.MultiBinary(self.prb_instance.n_machines),
            'jobs': gym.spaces.MultiBinary(self.prb_instance.n_jobs),
            'do_nothing': gym.spaces.Discrete(2)})
        return action_space


    def get_observation_space(self) -> gym.spaces.Dict:
        """ The observation space consists of:
            - Machine state: n_machines x 2 matrix each row is (current job, finish time)
                            - If the machine is idle, the finish time is 0
                            - If the machine is working on a job, the state is (job_id, finish_time)
            - Job state: dictionary {job_id: (next_operation_idx, is_started)}
                            - next_operation_idx: index of the next operation to be executed (if -1, the job is finished)
                            - is_started (0, 1): 1 if the job has started, 0 otherwise
            - Operation state: np.array of shape (n_jobs, n_machines) with values -1, 0, 1
                            - Indexed by job and machine
                            - operation_state[job_id, machine_id] = -1: machine is not used for the job
                            - operation_state[job_id, machine_id] = 1: Operation is scheduled or has finished successfully
                            - operation_state[job_id, machine_id] = 0: Operation is waiting to be scheduled
            - Current time: scalar value
        """
        # Machine state
        machine_state_low = np.array([0, 0])
        machine_state_high = np.array([self.prb_instance.n_jobs, np.inf])
        machine_state_space = gym.spaces.Box(
            low=machine_state_low,
            high=machine_state_high,
            shape=(self.prb_instance.n_machines, 2),
            dtype=np.float32)

        # Job state
        job_state_space = gym.spaces.Dict({
            str(job_id): gym.spaces.Tuple((
                    gym.spaces.Box(low=-1, high=self.prb_instance.n_ops[job_id] - 1, shape=(), dtype=np.int32),
                    gym.spaces.Discrete(2)))
            for job_id in range(self.prb_instance.n_jobs)})

        # Operation state
        operation_state_space = gym.spaces.Box(low=-1, high=1, 
                                               shape=(self.prb_instance.n_jobs, self.prb_instance.n_machines), 
                                               dtype=np.int32)

        # Current time
        current_time_space = gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32)

        # Combine into a single observation space
        observation_space = gym.spaces.Dict({
            'machine_state': machine_state_space,
            'job_state': job_state_space,
            'operation_state': operation_state_space,
            'current_time': current_time_space})

        return observation_space

    def reset(self) -> Tuple[dict, dict]:
        """ Reset the environment to initial state
            Initial machine state:
                - All machines are idle
                - All machines are assigned to the initial job
                - Thus the machine state is a n_machines x 2 matrix with (init_job, 0) for all machines
            Initial job state:
                - No jobs have started and the next operation is the first operation
            Initial operation state:
                - All (job, machine) not used are -1
                - All other operations are 0 (waiting to be scheduled) 
            Current time is 0

            returns:
                state (dict): dictionary with the machine state, job state and current time
                info (dict): additional information
        """
        info = {}
        # Initialize the machine state
        current_jobs = np.array(self.prb_instance.machines_initial_state) # All machines are assigned to the initial job
        finish_times = np.zeros(self.prb_instance.n_machines)  # Finish time is 0
        machine_state = np.stack((current_jobs, finish_times), axis=1)

        # Initialize the job state
        job_state = {str(job_id): (0, 0) for job_id in range(self.prb_instance.n_jobs)}

        # Initialize the operation state
        operation_state = np.zeros((self.prb_instance.n_jobs, self.prb_instance.n_machines))
        non_used_idxs = np.where(self.prb_instance.op_from_j_m == -1)
        operation_state[non_used_idxs] = -1

        # Current time is 0
        current_time = 0

        self.state = {
            'machine_state': machine_state,
            'job_state': job_state,
            'operation_state': operation_state,
            'current_time': current_time}
        return self.state, info

    def step(self, action) -> Tuple[Tuple[int, int], float, bool, dict]:
        """ Execute the actions in the environment.
            1) Assign the selected job to the selected machine (act on the environment)
            2) Update the environment state based on the selected actions
            3) Compute the reward for the selected actions
            4) Check if the episode is finished
            5) Return the new state, reward, done and info
            args:
                action: dict mult
        """
        # Initializations
        done = False
        info = {}

        # Compute the reward before executing the action
        reward_t = self.compute_reward()
        is_valid_action = True

        if action['do_nothing'] == 1:
            # Get the selected actions
            machine_ids = np.where(action['machines'] == 1)[0]
            job_ids = np.where(action['jobs'] == 1)[0]
            actions = [(machine_id, job_id) for machine_id in machine_ids for job_id in job_ids]

            # Execute the actions
            for action in actions:
                if not self.is_action_valid(action):
                    is_valid_action = False
                    continue
                self.start_operation(action)

        # Set the new current time to the next operation in the environment that will finish
        self.advance_to_next_event()

        # Check if the episode is finished
        done = self.is_done()
        reward_t1 = self.compute_reward(is_valid_action)
        reward = reward_t1 - reward_t

        return self.state, reward, done, info
    
    def compute_reward(self, is_valid_action=True) -> float:
        """ Compute the reward for the current state R_s(t)
            To compute the reward of an action R(a_t+1) = R_s(t+1) - R_s(t)
            R(t) is an approximation of the objective function value at time t.
            R(t) = W_E * E(t) + W_T * T(t) + W_F * F(t)
            where:
                - E(t) is an approximation of the earliness at time t
                - T(t) is an approximation of the tardiness at time t
                - F(t) is an approximation of the flow time at time t
                - W_E, W_T, W_F are weights for the earliness, tardiness and flow time respectively        
        """
        if not is_valid_action:
            return -np.inf # Invalid action should be penalized
        reward = 0
        W_E = self.prb_instance.jobs['earliness_penalty'].values
        W_T = self.prb_instance.jobs['tardiness_penalty'].values
        W_F = self.prb_instance.jobs['flow_time_penalty'].values
        for job_id, (w_e, w_t, w_f) in enumerate(zip(W_E, W_T, W_F)):
            mu_s = self.expected_completion(job_id)
            reward += w_e * self.early_approx(job_id, mu_s) + w_t * self.approx_tardy(job_id, mu_s)
            reward += w_f * self.flow_approx(job_id, mu_s)
        return reward

    def expected_completion(self, job_id) -> float:
        """ Compute the expected completion time of a job E(t) at time t.
            as the sum of the remaining processing times and mean setup times.
            E(t) = sum(r_i + mu_s) + t
                - r_i: processing time of operation i
                - mu_s: mean setup time for all operations
        """
        mean_setup_time = np.mean(self.prb_instance.df_setup_job['time_h'].values)
        remaining_processing_time = 0
        next_op_idx = self.state['job_state'][str(job_id)][0]
        num_ops = self.prb_instance.n_ops[job_id]
        for op in self.prb_instance.lst_job[job_id]:
            for i in range(next_op_idx, num_ops):
                remaining_processing_time += op['processing_time'] + mean_setup_time
        return self.state['current_time'] + remaining_processing_time
            
    def approx_tardy(self, job_id, expected_completion_time) -> float:
        """ Compute the tardiness approximation T(t) at the current time t.
            T(t) = max(0, expected_completion_time - due_date)
        """
        due_date = self.prb_instance.jobs['due_date'][job_id]
        approx_tardiness = max(0, expected_completion_time - due_date)
        return approx_tardiness

    def flow_approx(self, job_id, expected_completion_time) -> float:
        """ Compute the flow time approximation F(t) at the current time t.
            F(t) = expected_completion_time - release_date
        """
        release_date = self.prb_instance.jobs['release_date'][job_id]
        approx_flow_time = expected_completion_time - release_date
        return approx_flow_time

    def early_approx(self, job_id, expected_completion_time) -> float:
        """ Compute the earliness approximation E(t) at the current time t.
            E(t) = max(0, due_date - expected_completion_time)
        """
        due_date = self.prb_instance.jobs['due_date'][job_id]
        approx_earliness = max(0, due_date - expected_completion_time)
        return approx_earliness

    def is_done(self) -> bool:
        """ Check if the episode is finished.
            The episode is finished when all jobs have been completed.
        """
        for job_id in range(self.prb_instance.n_jobs):
            if self.state['job_state'][str(job_id)][0] != -1:
                return False
        return True

    def advance_to_next_event(self):
        """ Marks the next time event in the environment as the time when the next operation will finish.
            When an operation is finished it will be marked as a failure with a probability of failure_prob.
            In case of no failures the following is changed in the environment:
                - Machine state:
                    - The machine is updated to idle by setting the finish time to 0
                - Job state:
                    - No changes are made as this is done when the operation is started
                - Operation state:
                    - No changes are made as this is done when the operation is started
            In case of failure see self.restart_operation
        """
        finish_times = self.state['machine_state'][:, 1]
        next_finish_time = np.min(finish_times[finish_times > 0])

        # Update the current time
        self.state['current_time'] = next_finish_time

        # Get the machines with the next finish time
        next_idle_machines = np.where(self.state['machine_state'][:, 1] == next_finish_time)[0]

        # Update the environment state
        for machine_id in next_idle_machines:
            # Update machine state to idle
            self.state['machine_state'][machine_id][1] = 0

            if np.random.rand() < self.failure_prob: # Failure occurs
                job_id = self.state['machine_state'][machine_id][0] 
                self.restart_operation(machine_id, job_id)
                            
    def restart_operation(self, machine_id, job_id):
        """ Restart an operation on a machine after a failure.
            In case of a failure the following is changed in the environment:
                - Machine state:
                    - No changes are made as the machine is already idle
                - Job state:
                    - The next operation index is decremented by 1
                    - If the operation index was -1 the index will be set the last operation index
                - Operation state:
                    - The operation is marked as not scheduled i.e 0
        """
        # Update the job state (next operation index)
        next_operation_idx = self.state['job_state'][str(job_id)][0]
        if next_operation_idx == -1:
            next_operation_idx = self.prb_instance.n_ops[job_id] - 1
        else:
            next_operation_idx -= 1
        self.state['job_state'][str(job_id)][0] = next_operation_idx

        # Update the operation state
        self.state['operation_state'][job_id, machine_id] = 0

    def start_operation(self, action):
        """ Start an operation on a machine.
            When an action is started the following is changed in the environment:
                - Machine state:
                    - The current job is updated to the selected job
                    - The finish time is updated to the time when the operation will finish
                - Job state:
                    - The next operation index is updated to the next operation
                        - If the previous operation was the last operation, the next operation is -1
                    - The job is marked as started
                - Operation state:
                    - The operation is marked as scheduled i.e 1
        """
        machine_id, job_id = action

        # Get the setup time and process time for the operation
        current_time = self.state['current_time']
        previous_job = self.state['machine_state'][machine_id][0]
        setup_time = self.prb_instance.get_setup(machine_id, previous_job, job_id)
        process_time = self.prb_instance.lst_job[job_id][0]['processing_time']
        finish_time = current_time + process_time + setup_time

        # Update the machine state (current job, finish time)
        self.state['machine_state'][machine_id][0] = job_id 
        self.state['machine_state'][machine_id][1] = finish_time 


        job_state = self.state['job_state'][str(job_id)]
        if job_state[0] == self.prb_instance.n_ops[job_id] - 1: # Last operation
            self.state['job_state'][str(job_id)][0] = -1
        else:
            self.state['job_state'][str(job_id)][0] += 1

        if job_state[1] == 0: # Start the job
            self.state['job_state'][str(job_id)][1] = 1

        # Update the operation state
        self.state['operation_state'][job_id, machine_id] = 1

    def is_action_valid(self, action):
        """ Check if the action is valid.
            An action is valid if:
                - The machine is idle
                - The operation is eligible
                - The operation is assigned to the correct machine
                - The job has not finished
                - The proposed operation is the next operation for the job
        """
        machine_id, job_id = action
        if self.state['machine_state'][machine_id][1] != 0: # Machine is not idle
            return False 
        if self.state['operation_state'][job_id, machine_id] != 0: # Operation is not eligible
            return False
        if self.prb_instance.eligible_machines[(job_id, machine_id)] != machine_id: # Operation is not assigned to the correct machine
            return False
        next_operation_idx = self.state['job_state'][str(job_id)][0]
        if next_operation_idx == -1: # No next operation means the job is finished
            return False
        proposed_operation_idx = self.prb_instance.op_from_j_m[job_id, machine_id]
        if proposed_operation_idx != next_operation_idx: # Operation is not the next operation
            return False
        return True
        
    def simulate_scheduling(self):
        """ Simulate the scheduling of the environment. """
        for _ in range(100):
            action = self.action_space.sample()
            self.step(action)
        raise NotImplementedError
    
    def render(self):
        # Render the environment to the screen using plot operations forest
        self.prb_instance.plot_operations_forest()
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError
