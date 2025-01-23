import numpy as np
from .scheduler import JobShopScheduler

class StaticAgest(JobShopScheduler):
    def __init__(self,priority_rule: str):
        super(StaticAgest,self).__init__()
        self.priority_rule= priority_rule
    
    def get_schedule(self, env):
        """ Schedule the jobs using static rules """
        max_iter = 1000                                                           
        while self.all_jobs_scheduled(env):
            if self.priority_rule == 'edd':
                due_dates = env.prb_instance.df_jobs['due_date'].values
                priorities = np.argsort(due_dates)
            elif self.priority_rule == 'lpt':
                processing_times = env.prb_istance.list_job.values
                priorities = sorted(range(len(processing_times)), key=lambda x: processing_times[x], reverse=True)
            elif self.priority_rule == 'spt':
                priorities = env.prb_istance.list_job.values
                priorities = sorted(range(len(priorities)), key=lambda x: priorities[x])
            elif self.priority_rule == 'wspt':
                processing_times = env.prb_istance.list_job.values
                weights = env.prb_istance.df_jobs['tardiness_penalty'].values
                priorities = sorted(range(len(processing_times)), key=lambda x: processing_times[x] / weights[x])
                
            idle_machine_idxs = np.where(env.state['machine_state'][:, 1] == 0)[0]          #List of idle machines
            
            for machine_idx in idle_machine_idxs:
                for job_idx in priorities:                                                  #funzione che calcola le priority
                    if self.is_job_ready_for_scheduling(env, job_idx, machine_idx):
                        op_idx = env.state['job_state'][job_idx, 0]
                        env = self.schedule_operation(env, job_idx, op_idx, machine_idx)
                        break
                # Fast forward the environment to the next event
            env = self.fast_forward_to_next_event(env)
            max_iter -= 1                                                                   #aggiorna priorities attraverso funzione
            if max_iter == 0:
                print("Max iterations reached")
                return None
        return env.state['schedule_state']
    

class DynamicAgents(JobShopScheduler):
    def __init(self,priority_rule: str, current_time: float):
        super(DynamicAgents,self).__init__()
        self.priority_rule= priority_rule
        self.current_time = current_time        #Passing env.state['current_time] from outside??

    def get_schedule(self, env):
        """ Schedule the jobs using the dynamic rules """
        max_iter = 1000
        if self.priority_rule == 'atcs':
            priorities= self.ATCS(env)
        elif self.priority_rule == 'msf':
            priorities= self.MSF(env)
        while self.all_jobs_scheduled(env):

            idle_machine_idxs = np.where(env.state['machine_state'][:, 1] == 0)[0] 
            
            for machine_idx in idle_machine_idxs:
                for job_idx in priorities:                                                  #funzione che calcola le priority
                    if self.is_job_ready_for_scheduling(env, job_idx, machine_idx):
                        op_idx = env.state['job_state'][job_idx, 0]
                        env = self.schedule_operation(env, job_idx, op_idx, machine_idx)
                        break
            # Fast forward the environment to the next event
            env = self.fast_forward_to_next_event(env)
            if self.priority_rule == 'atcs':
                priorities = self.ATCS(env)
            elif self.priority_rule == 'mfs':
                priorities= self.MSF(env)
            max_iter -= 1                                                                   #aggiorna priorities attraverso funzione
            if max_iter == 0:
                print("Max iterations reached")
                return None
        return env.state['schedule_state']
    
    def ATCS(self,env):
        processing_time = env.prb_istance.list_job.values  
        weights = env.prb_istance.df_jobs['tardiness_penalty'].values  
        due_dates = env.prb_instance.df_jobs['due_date'].values
        
        # Compute the schedule dynamically
        schedule = []
        timer = self.current_time
        jobs_to_schedule = list(range(len(processing_time)))

        while len(jobs_to_schedule) > 0:
            # Calculate ATC for the remaining jobs, considering their current completion time
            atc = [
                weights[job] * max(0, (timer + processing_time[job]) - due_dates[job])
                for job in jobs_to_schedule
            ] # ATC = W * max(0, (C + P) - D)
            
            # Select the job with the highest ATC
            next_job = max(jobs_to_schedule, key=lambda x: atc[x])

            # Schedule the selected job
            schedule.append(next_job)
            jobs_to_schedule.remove(next_job)

            timer += processing_time[next_job]

        return schedule
    
    def MSF(self,env):
        processing_time = env.prb_istance.list_job.values  
        due_dates = env.prb_instance.df_jobs['due_date'].values 
        
        # Schedule the jobs dynamically
        schedule = []
        timer = self.current_time
        jobs_to_schedule = list(range(len(processing_time)))

        while len(jobs_to_schedule) > 0:
            # Calculate slack for each remaining job
            slack = [
                due_dates[job] - (timer + processing_time[job])  # Slack = D - (C + P)
                for job in jobs_to_schedule
            ]
            
            # Select the job with the smallest slack time
            next_job = min(jobs_to_schedule, key=lambda x: slack[x])

            # Schedule the selected job
            schedule.append(next_job)
            jobs_to_schedule.remove(next_job)

            # Update the current time after scheduling this job
            timer += processing_time[next_job]

        return schedule