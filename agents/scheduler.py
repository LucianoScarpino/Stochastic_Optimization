import numpy as np

class JobShopScheduler(object):
    """ Parent class for priority ruled scheduling agents (see agents/agents.py)
        methods:
            is_job_ready_for_scheduling: Check if a proposed job can be scheduled on a machine
            schedule_operation: Schedule an operation for a job on a machine
            fast_forward_to_next_event: Fast forward the environment to the time of next idle machine
            all_jobs_scheduled: Check if all the jobs are scheduled
    """
    def __init__(self):
        pass

    def is_job_ready_for_scheduling(self, env, job_idx, machine_idx,) -> bool:
        """ Check if a proposed job can be scheduled on a machine """
        op_idx = env.state['job_state'][job_idx, 0]
        if op_idx == -1: #No pending operations                             
            return False
        if env.state['job_state'][job_idx, 1] == 1: # Job is already running
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
        """ Fast forward the environment to the time of next idle machine,
            udpate the job and machine states accordingly.
        """
        machine_idle_times = env.state['machine_state'][:, 1]
        if np.all(machine_idle_times == 0): # All machines are idle
            return env
        time_until_next_idle = np.min(machine_idle_times[machine_idle_times > 0])
        next_idle_machines = np.where(machine_idle_times == time_until_next_idle)[0]

        for machine_id in next_idle_machines:
            # Update the job state
            job_idx = env.state['machine_state'][machine_id, 0] # Get the job index of completed job
            if env.state['job_state'][job_idx, 0] == env.prb_instance.n_ops[job_idx] - 1: # Current operation is the last operation
                env.state['job_state'][job_idx, 0] = -1
            else:
                env.state['job_state'][job_idx, 0] += 1 # Increment the operation index
            env.state['job_state'][job_idx, 1] = 0 # Set the job to idle

            # Update the machine and schedule state
            env.state['machine_state'][machine_id, 1] = 0 # Set the machine to idle

        # Fast forward the current time and remaining times of the working machines
        env.state['current_time'] += time_until_next_idle
        env.state['machine_state'][:, 1] = np.maximum(0, env.state['machine_state'][:, 1] - time_until_next_idle)
        return env

    def all_jobs_scheduled(self, env) -> bool:
        """ Check if all the jobs are scheduled """
        for job_idx in range(env.prb_instance.n_jobs):
            if env.state['job_state'][job_idx, 0] != -1:
                return True
        return False