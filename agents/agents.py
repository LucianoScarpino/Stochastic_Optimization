import numpy as np
from .scheduler import JobShopScheduler

class StaticAgent(JobShopScheduler):
    """ Scheduling agent that schedules remaining operations based on static rules
        and current state of the environment (see envs/shopFloor.py).
        args:
            priority_rule: str: The priority rule to be used for scheduling
        methods:
            compute_processing_times: Compute the processing times of the jobs
            get_schedule: Schedule the jobs using static rules
            get_priority_list: Return the priority list based on the static rule
    """

    def __init__(self, priority_rule: str):
        super(StaticAgent,self).__init__()
        assert priority_rule in ['edd', 'lpt', 'spt', 'wspt'], "Invalid priority rule"
        self.priority_rule= priority_rule
    
    def compute_processing_times(self, env):
        """ Compute the processing times of the jobs """
        processing_times = []
        for job in env.prb_instance.lst_job:
            next_process_time = 0
            for op_idx in range(len(job)):
                next_process_time += job[op_idx]['processing_time']
            processing_times.append(next_process_time)
        return processing_times
    
    def get_schedule(self, env):
        """ Schedule the jobs using static rules """   
        priority_list = self.get_priority_list(env)   
        while self.all_jobs_scheduled(env):                
            idle_machine_idxs = np.where(env.state['machine_state'][:, 1] == 0)[0]          #List of idle machines
            for machine_idx in idle_machine_idxs:
                for job_idx in priority_list:                                                  
                    if self.is_job_ready_for_scheduling(env, job_idx, machine_idx):
                        op_idx = env.state['job_state'][job_idx, 0]
                        env = self.schedule_operation(env, job_idx, op_idx, machine_idx)
                        break
            env = self.fast_forward_to_next_event(env)
        return env.state['schedule_state']
    
    def get_priority_list(self, env) -> list:
        """ Return the priority list based on the static rule """
        processing_times = self.compute_processing_times(env)
        if self.priority_rule == 'edd':
            due_dates = env.prb_instance.df_jobs['due_date'].values
            priority_list = np.argsort(due_dates)
        elif self.priority_rule == 'lpt':
            priority_list = sorted(range(len(processing_times)), 
                                key=lambda x: processing_times[x], reverse=True)
        elif self.priority_rule == 'spt':
            priority_list = sorted(range(len(processing_times)), 
                                key=lambda x: processing_times[x])
        elif self.priority_rule == 'wspt':
            weights = env.prb_instance.df_jobs['tardiness_penalty'].values
            priority_list = sorted(range(len(processing_times)), 
                                key=lambda x: processing_times[x] / weights[x])
        return priority_list
    
class DynamicAgent(JobShopScheduler):
    """ Scheduling agent that schedules remaining operations based on dynamic rules
        and current state of the environment (see envs/shopFloor.py).
        args:
            priority_rule: str: The priority rule to be used for scheduling
        methods:
            get_schedule: Schedule the jobs using dynamic rules
            get_atcs_priority_list: Return the priority list based on the ATCS rule
            get_msf_priority_list: Return the priority list based on the MSF rule
            get_processing_times: Compute the processing times of the remaining jobs
    """
    def __init__(self, priority_rule: str):
        super(DynamicAgent,self).__init__()
        assert priority_rule in ['atcs', 'msf'], "Invalid priority rule"
        self.priority_rule= priority_rule
        if self.priority_rule == 'atcs':
            self.get_priority_list = self.get_atcs_priority_list
        elif self.priority_rule == 'msf':
            self.get_priority_list = self.get_msf_priority_list
        
    def get_schedule(self, env):
        """ Schedule the jobs using the dynamic rules """
        while self.all_jobs_scheduled(env):
            priority_list = self.get_priority_list(env) # Update the priority list dynamically
            idle_machine_idxs = np.where(env.state['machine_state'][:, 1] == 0)[0] 
            for machine_idx in idle_machine_idxs:
                for job_idx in priority_list:                                                  
                    if self.is_job_ready_for_scheduling(env, job_idx, machine_idx):
                        op_idx = env.state['job_state'][job_idx, 0]
                        env = self.schedule_operation(env, job_idx, op_idx, machine_idx)
                        break
            env = self.fast_forward_to_next_event(env)
        return env.state['schedule_state']
    
    def get_processing_times(self, env) -> np.ndarray:
        """ Compute the processing times of the remaining jobs """
        processing_times = np.zeros(env.prb_instance.n_jobs)
        for job_idx in range(env.prb_instance.n_jobs):
            op_idx = env.state['job_state'][job_idx, 0]
            if op_idx == -1:                            # Job has no pending operations
                continue
            proceess_time = 0
            for i in range(op_idx, env.prb_instance.n_ops[job_idx]):
                proceess_time += env.prb_instance.lst_job[job_idx][i]['processing_time']
            processing_times[job_idx] = proceess_time
        return processing_times

    def get_atcs_priority_list(self, env) -> list:
        """ Return the priority list based on the ATCS rule """
        processing_time = self.get_processing_times(env)
        weights = env.prb_instance.df_jobs['tardiness_penalty'].values  
        due_dates = env.prb_instance.df_jobs['due_date'].values

        priority_list = []
        timer = env.state['current_time']
        jobs_to_schedule = list(range(len(processing_time)))

        while len(jobs_to_schedule) > 0:
            atc = {
                job: weights[job] * max(0, (timer + processing_time[job]) - due_dates[job])
                for job in jobs_to_schedule}
            next_job = max(atc, key=atc.get)

            # Schedule the selected job
            priority_list.append(next_job)
            jobs_to_schedule.remove(next_job)

            timer += processing_time[next_job]

        return priority_list
    
    def get_msf_priority_list(self, env) -> list:
        """ Return the priority list based on the MSF rule """
        processing_time = self.get_processing_times(env)
        due_dates = env.prb_instance.df_jobs['due_date'].values 
        
        # Schedule the jobs dynamically
        priority_list = []
        timer = env.state['current_time']
        jobs_to_schedule = list(range(len(processing_time)))

        while len(jobs_to_schedule) > 0:
            # Calculate slack for each remaining job
            slack = {
                job: due_dates[job] - (timer + processing_time[job])  # Slack = D - (C + P)
                for job in jobs_to_schedule
            }
            next_job = min(slack, key=slack.get) 
            # Schedule the selected job
            priority_list.append(next_job)
            jobs_to_schedule.remove(next_job)
            timer += processing_time[next_job]
        return priority_list