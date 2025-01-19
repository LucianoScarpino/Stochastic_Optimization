import numpy as np

class JobShopScheduler(object):
    """ Parent class for each priority-based scheduler. The environment used is defined in (envs/shopFloor.py) 
        Class methods:
            - is_job_ready_for_scheduling(env, job_idx, machine_idx) -> bool: Check if a job can be scheduled on a machine
            - schedule_operation(env, job_idx, op_idx, machine_idx) -> env: Schedule an operation for a job on a machine
            - fast_forward_to_next_event(env) -> env: Fast forward the environment to the time of next idle machine
            - all_jobs_scheduled(env) -> bool: Check if all the jobs are scheduled
    """

    def __init__(self):
        pass

    def is_job_ready_for_scheduling(self, env, job_idx, machine_idx,) -> bool:
        """ Check if a job can be scheduled on a machine
            args:
                env: The environment (see envs/shopFloor.py)
                job_idx: int: The index of the job
                machine_idx: int: The index of the machine
            returns:
                bool: True if the job can be scheduled, False otherwise
        """
        op_idx = env.state['job_state'][job_idx, 0]
        if op_idx == -1: # The job is finished
            return False
        if env.state['job_state'][job_idx, 1] == 1: # Job has an operation running
            return False
        if env.prb_instance.eligible_machines[(op_idx, job_idx)] != machine_idx: # Machine is not eligible
            return False
        return True
    
    def schedule_operation(self, env, job_idx, op_idx, machine_idx):
        """ Schedule an operation for a job on a machine
            args:
                env: The environment (see envs/shopFloor.py)
                job_idx: int: The index of the job
                op_idx: int: The index of the operation
                machine_idx: int: The index of the machine
            returns:
                env: The updated environment
        """
        # Get the current job on the machine and the current time
        current_job_on_machine = env.state['machine_state'][machine_idx, 0]
        current_time = env.state['current_time']

        # Get the processing time
        processing_time = env.prb_instance.lst_job[job_idx][op_idx]['processing_time']
        setup_time = env.prb_instance.get_setup(machine_idx, current_job_on_machine, job_idx)
        remaining_time = setup_time + processing_time

        # Update the environment state
        env.state['job_state'][job_idx] = [op_idx, 1]
        env.state['machine_state'][machine_idx] = [job_idx, remaining_time]
        env.state['schedule_state'][job_idx, op_idx] = [current_time, remaining_time, machine_idx, 0]
        
        return env

    def fast_forward_to_next_event(self, env):
            """ Fast forward the environment to the time of next idle machine. Following changes are made to the environment:
                - Job state: Jobs with an operation that is finished are updated:
                    - current_op_idx:
                        - If the job is finished, the current operation index is set to -1
                        - If the job is not finished, the operation index is incremented by 1
                    - is_running is set to 0
                - Machine state:
                    - Remaing time of the finished machines is set to 0
                    - Remaining time of working machines is decremented by the time until the next idle machine
                - Schedule state: 
                    - The status variable of the finished operations is set to 1
                - Current time:
                    - The current time is updated to the time of the next idle machine
                args:
                    env: The environment (see envs/shopFloor.py)
                returns:
                    env: The updated environment
            """
            machine_idle_times = env.state['machine_state'][:, 1]
            if np.all(machine_idle_times == 0): # All machines are idle
                return env
            time_until_next_idle = np.min(machine_idle_times[machine_idle_times > 0])
            next_idle_machines = np.where(machine_idle_times == time_until_next_idle)[0]

            for machine_id in next_idle_machines:

                # Update the job state
                job_idx = env.state['machine_state'][machine_id, 0] # Get the job index of completed job
                if env.state['job_state'][job_idx, 0] == env.prb_instance.n_ops[job_idx] - 1: # Check if the job is finished
                    env.state['job_state'][job_idx, 0] = -1
                else:
                    env.state['job_state'][job_idx, 0] += 1 # Increment the operation index
                env.state['job_state'][job_idx, 1] = 0 # Set the job to idle

                # Update the machine and schedule state
                env.state['machine_state'][machine_id, 1] = 0 # Set the remaining time to 0

            env.state['current_time'] += time_until_next_idle
            env.state['machine_state'][:, 1] = np.maximum(0, env.state['machine_state'][:, 1] - time_until_next_idle)
            return env

    def all_jobs_scheduled(self, env) -> bool:
        """ Check if all the jobs are scheduled """
        for job_idx in range(env.prb_instance.n_jobs):
            if env.state['job_state'][job_idx, 0] != -1:
                return True
        return False

