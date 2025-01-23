import numpy as np
from .scheduler import JobShopScheduler

class EddAgent(JobShopScheduler):
    def __init__(self):
        super(EddAgent, self).__init__()

    def get_schedule(self, env):
        """ Schedule the jobs using the earliest due date rule """
        max_iter = 1000                                                                     #variabile priorità che tiene traccia della priorità fuori dal while
        while self.all_jobs_scheduled(env):
            due_dates = env.prb_instance.df_jobs['due_date'].values
            priorities = np.argsort(due_dates)
            idle_machine_idxs = np.where(env.state['machine_state'][:, 1] == 0)[0] #List of idle machines
            
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