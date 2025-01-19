import numpy as np
from .scheduler import JobShopScheduler

class EddAgent(JobShopScheduler):
    def __init__(self):
        super(EddAgent, self).__init__()

    def get_schedule(self, env):
        """ Schedule the jobs using the earliest due date rule """
    
        while self.all_jobs_scheduled(env):
            due_dates = env.prb_instance.df_jobs['due_date'].values
            priorities = np.argsort(due_dates)
            idle_machine_idxs = np.where(env.state['machine_state'][:, 1] == 0)[0] #List of idle machines
            
            for machine_idx in idle_machine_idxs:
                for job_idx in priorities:
                    if self.is_job_ready_for_scheduling(env, job_idx, machine_idx):
                        op_idx = env.state['job_state'][job_idx, 0]
                        env = self.schedule_operation(env, job_idx, op_idx, machine_idx)
                        break
            # Fast forward the environment to the next event
            env = self.fast_forward_to_next_event(env)
        return env.state['schedule_state']