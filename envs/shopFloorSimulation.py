from .shopFloorEnvironment import ShopFloorEnvironment
from .shopFloorGantt import GanttCharts
import numpy as np

class ShopFloorSimulation(ShopFloorEnvironment):
    """ Class for the shop floor scheduling environment
        args:
            prb_instance: The problem instance (see data_interfaces.py)
            agent: The agent to use for scheduling (see agents/)
            priority_rule: The priority rule to use for scheduling
            image_folder: The folder to save the images
            failure_prob: The probability of failure for each operation
        attributes:
            gantt_plotter: The Gantt chart plotter (see envs/shop_floor_gantt.py)
    """
    def __init__(self, prb_instance, agent, image_folder='images', failure_prob=0.0):
        super(ShopFloorSimulation, self).__init__(prb_instance)
        self.gantt_plotter = GanttCharts(img_dir=image_folder)
        self.priority_rule = agent.priority_rule
        self.agent = agent
        self.failure_prob = failure_prob
        self.state = self.get_state_space()
        self.state = self.reset()
    
    def simulate_scheduling(self, schedule: np.ndarray, plot_gantt=True, caption: str = None) -> float:
        """ Simulate the scheduling of the jobs.
        Args:
            schedule (n_jobs, max_ops, 4): The initial schedule
            plot_progress (bool): If True, plot the progress of the simulation
            caption (str): Extra information to add to the title of the image
        Returns:
            obj_func (float): The objective function value of the final schedule
        """
        num_epochs = 0
        self.initialize_simulation(schedule)
        while not self.is_finished():
            num_epochs += 1
            next_time_epoch, event_type = self.get_next_time_epoch()
            self.current_time = next_time_epoch
            if event_type == 'start':
                self.start_operations(next_time_epoch)
            elif event_type == 'finish':
                succes, time_step = self.finish_operations(next_time_epoch)
                self.state['current_time'] += time_step # Update the current time
                if not succes:
                    self.reschedule(next_time_epoch)
            self.state['current_time'] = next_time_epoch
            if plot_gantt:
                self.gantt_plotter.generate_gantt_chart(self, f"epoch_{num_epochs}", caption)         
        obj_func = self.compute_objective_function()
        return obj_func
    
    def reschedule(self, time_stamp) -> None:
        """ Reschedule the operations if a machine fails """
        old_job_state = self.state['job_state'].copy()
        old_machine_state = self.state['machine_state'].copy()

        pending_ops = self.state['schedule_state'][:, :, 3] == 0
        self.state['schedule_state'][pending_ops] = [0, 0, 0, 0]        
        new_schedule = self.agent.get_schedule(self)                                

        # Restore the job state and machine state
        self.state['current_time'] = time_stamp
        self.state['job_state'] = old_job_state
        self.state['machine_state'] = old_machine_state
        self.state['schedule_state'] = new_schedule
        return None

    def compute_objective_function(self) -> float:
        """ Compute the objective function of the final schedule.
            Returns:
                obj_func (float): The objective function value of the schedule
        """
        obj_func = 0
        n_ops = [ops - 1 for ops in self.prb_instance.n_ops]
        start_times = self.state['schedule_state'][:, 0, 0]

        completions_times = []
        for job_idx in range(self.prb_instance.n_jobs):
            start_n = self.state['schedule_state'][job_idx, n_ops[job_idx], 0]    # Start time of the last operation
            duration_n = self.state['schedule_state'][job_idx, n_ops[job_idx], 1] # Duration of the last operation
            completions_times.append(start_n + duration_n)

        for job_idx, (c_j,s1) in enumerate(zip(completions_times, start_times)):
            job_data = self.prb_instance.df_jobs
            d_j = job_data['due_date'][job_idx]
            E_j = max(0, d_j - c_j)
            T_j = max(0, c_j - d_j)
            
            w_E = job_data['earliness_penalty'][job_idx]
            w_T = job_data['tardiness_penalty'][job_idx]
            w_F = job_data['flow_time_penalty'][job_idx]
            obj_func += w_E * E_j + w_T * T_j + w_F * (c_j - s1)
        return obj_func

    def finish_operations(self, next_time_epoch: float) -> tuple[bool , int]:
        """ Determine if the operations are successful and update the environment.
            args:
                next_time_epoch (float): Next time a machine finishes
            Returns:
                succes (bool): True if the operation was successfully completed, False otherwise
                time_step (float): The time step until the next event
        """
        succes = True
        # Find the machines and corresponding operations that are finishing at the next time epoch
        machine_idxs = np.where(self.state['machine_state'][:, 1] == next_time_epoch - self.state['current_time'])[0]
        job_idxs = self.state['machine_state'][machine_idxs, 0]
        op_idxs = self.state['job_state'][job_idxs, 0]

        for job_id, op_id in zip(job_idxs, op_idxs):
            # Mark the operation as completed
            if np.random.rand() > self.failure_prob: 
                self.state['job_state'][job_id, 1] = 0
                if op_id == self.prb_instance.n_ops[job_id] - 1: # No further operations
                    self.state['job_state'][job_id, 0] = -1
                else:
                    self.state['job_state'][job_id, 0] += 1
                
                self.state['schedule_state'][job_id, op_id, 3] = 2
            else:
                # Mark the operation as failed 
                self.state['schedule_state'][job_id, op_id, 3] = 0
                self.state['job_state'][job_id, 1] = 0
                succes = False

        # Reset the machine state
        time_step = next_time_epoch - self.state['current_time']
        self.state['machine_state'][:, 1] = np.maximum(0, self.state['machine_state'][:, 1] - time_step)
        return succes, time_step

    def is_finished(self) -> bool:
        """ Check if there are more pending operations """
        if np.all(self.state['job_state'][:, 0] == -1):
            return True

    def start_operations(self, next_time_epoch: float) -> None:
        """ Start operations starting at the next time epoch """
        # Find the operations that are starting at the next time epoch
        pending_ops = self.state['schedule_state'][:, :, 3] == 0
        starting_jobs = np.where(self.state['schedule_state'][pending_ops, 0] == next_time_epoch)[0]
        job_idxs = np.where(pending_ops)[0][starting_jobs]
        op_idxs = self.state['job_state'][job_idxs, 0]

        durations = self.state['schedule_state'][job_idxs, op_idxs, 1] # Setups are included in the duration
        machine_idxs = [int(id) for id in self.state['schedule_state'][job_idxs, op_idxs, 2]]

        # Update the machine state (note the setup times are already included the operation duration)
        self.state['machine_state'][machine_idxs, 0] = job_idxs
        self.state['machine_state'][machine_idxs, 1] = durations

        # Update the environment
        self.state['job_state'][job_idxs, 1] = 1
        self.state['schedule_state'][job_idxs, op_idxs, 3] = 1

        return None

    def get_next_time_epoch(self) -> tuple[float, str]:
        """ Return the next time epoch and the type of event (start or finish),
            in case of a tie finish time has priority.
            returns:
                next_time_epoch: The next time epoch
                event_type: The type of event ['start', 'finish']
        """
        # Next start times        
        pending_ops = self.state['schedule_state'][:, :, 3] == 0
        pending_ops_start_times = self.state['schedule_state'][pending_ops, 0]
        
        # Next finish times
        non_idle_machines = self.state['machine_state'][:, 1] > 0
        non_idle_machines_durations = self.state['machine_state'][non_idle_machines, 1]
        if len(non_idle_machines_durations) == 0:
            next_finish_time = np.inf
        else:
            next_finish_time = np.min(non_idle_machines_durations) + self.state['current_time']

        if len(pending_ops_start_times) == 0:
            return next_finish_time, 'finish'
        next_start_time = np.min(pending_ops_start_times)
        
        # In case of a tie finish time has priority
        if next_start_time == next_finish_time or next_finish_time < next_start_time:
            return next_finish_time, 'finish'
        elif next_start_time < next_finish_time:
            return next_start_time, 'start'
        else:
            raise ValueError('Invalid time epoch')
        
    def initialize_simulation(self, schedule) -> None:
        """ Initialize the simulation at time 0 """
        self.state['job_state'][:, 0] = 0
        self.state['job_state'][:, 1] = 0
        self.state['machine_state'][:, 0] = self.prb_instance.machines_initial_state
        self.state['schedule_state'] = schedule
        self.state['current_time'] = 0
        return None