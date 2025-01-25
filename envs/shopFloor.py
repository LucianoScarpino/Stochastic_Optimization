import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from pathlib import Path
from PIL import Image
import os

class ShopFloor(gym.Env):
    """States of the environment:
        1) Job state np.ndarray(n_jobs, 2): Status of the jobs at the current time
            - job_state[job_idx]: [current_op_idx, is_running]
                - current_op_idx: The index of the current operation being processed (-1 if no operation is being processed)
                - is_running: 0 if the job is idle, 1 if the job is running

        2) Machine state np.ndarray(n_machines, 2): Status of the machines at the current time
            - machine_state[machine_idx]: [current_job_idx, remaining_time]
                - current_job_idx: The index of the job working on the machine
                - remaining_time: The remaining time for the current operation (0 if the machine is idle)
        
        3) Schedule state np.ndarray(n_jobs, n_machines, 4): The current schedule of the jobs
            - schedule[job_idx, machine_idx]: [start, duration, machine_idx, status]
                - start: If an operation is being processed, the start time of the operation
                - duration: If an operation is being processed, the remaining time for the operation
                - machine_idx: The index of the machine that the operation is scheduled
                - status: The status of the operation can take the following values:
                    - 0: Operation is waiting to be processed
                    - 1: Operation is being processed
                    - 2: Operation is successfully completed
                - If an job is not scheduled for a machine, the value is [-1, -1, -1, -1]
        
        4) Current time: The current time of the environment
        args:
            prb_instance: The problem instance (see data_interfaces.py)
            gant_plotter: The Gantt plotter object (see GanttCharts)
            image_folder: str, The folder to store the images of the Gantt charts
            failure_prob: float, The probability of a machine failure
            use_predifined_failures: bool, use a set of predefined failures
    """
    def __init__(self, prb_instance, gantt_plotter, agent, priority_rule,
                 image_folder='images', failure_prob=0.0, use_predifined_failures=False):
        self.prb_instance = prb_instance
        self.gantt_plotter = gantt_plotter
        self.priority_rule = priority_rule
        self.agent = agent
        self.image_folder = image_folder
        self.failure_prob = failure_prob
        if use_predifined_failures:
            # Binary dictionary mapping each operation (job_idx, machine_idx) to a succes variable
            self.failures = {
                (job_idx, machine_idx): np.random.rand() > failure_prob # Succes(true) or failure(false)
                for job_idx in range(prb_instance.n_jobs)
                for machine_idx in range(prb_instance.n_machines)
            }
        else:
            self.failures = None

        self.state = self.get_state_space()
        self.state = self.reset()

        Path(self.image_folder).mkdir(parents=True, exist_ok=True)  
    
    def simulate_scheduling(self, schedule, plot_gantt=True) -> float:
        """ Simulate the scheduling of the jobs.
        Args:
            schedule (np.ndarray): The schedule initial schedule of the jobs (n_jobs, n_operations, 4)
                - schedule[job_idx, op_idx]: [start, duration, machine_idx, status]
            plot_progress (bool): If True, plot the progress of the simulation
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
                self.render_gantt_chart(f"epoch_{num_epochs}", with_caption=True)               #usa questa se vuoi controllare che schedula lavora bene.
        obj_func = self.compute_objective_function()
        return obj_func
    
    def render_gantt_chart(self, img_name: str, with_caption: bool = True) -> None:
        """ Render the current state of the environment as a Gantt chart """
        caption = f"Failure probability: {self.failure_prob}, Priority rule: {self.priority_rule}"
        if not with_caption:
            caption = None
        self.gantt_plotter.generate_gantt_chart(self, img_name, caption)


    def reschedule(self, time_stamp) -> None:
        """ Reschedule the operations that are still pending """
        # Clear all pending operations from the schedule
        # Save the job state and machine state
        job_state_ = self.state['job_state'].copy()
        machine_state_ = self.state['machine_state'].copy()

        pending_ops = self.state['schedule_state'][:, :, 3] == 0
        self.state['schedule_state'][pending_ops] = [0, 0, 0, 0]        
        new_schedule = self.agent.get_schedule(self)                                #Il nostro codice entra qui

        # Restore the job state and machine state
        self.state['current_time'] = time_stamp
        self.state['job_state'] = job_state_
        self.state['machine_state'] = machine_state_
        self.state['schedule_state'] = new_schedule
        return None

    def compute_objective_function(self) -> float:
        """ Compute the objective function of the final schedule.
            obj(schedule) = sum_{j} w_E * E_j + w_T * T_j + w_F(C_j - S_1)
                - E_j: The earliness of job j (E_j = max(0, d_j - C_j))
                - T_j: The tardiness of job j (T_j = max(0, C_j - d_j))
                - w_E, w_T, w_F: The weights of earliness, tardiness, and makespan
                - d_j: The due date of job j
                - C_j: The completion time of last operation of job j
                - C_{I_n}: The completion time of the last job
                - S_1: The start time of the first job
            Returns:
                obj_func (float): The objective function value of the schedule
        """
        obj_func = 0
        n_ops = [ops - 1 for ops in self.prb_instance.n_ops]
        start_times = self.state['schedule_state'][:, 0, 0]
        # start_times = np.min(self.state['schedule_state'][:, 0, 0])

        completions_times = []
        for job_idx in range(self.prb_instance.n_jobs):
            start_n = self.state['schedule_state'][job_idx, n_ops[job_idx], 0] # Start time of the last operation
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
        """ If a machine finishes an operation, update the state of the environment.
            Returns:
                succes (bool): True if the operation was successfully completed, False otherwise
                time_step (float): The time step until the next event
        """
        succes = True
        # Find the machines that are finishing an operation
        machine_idxs = np.where(self.state['machine_state'][:, 1] == next_time_epoch - self.state['current_time'])[0]
        job_idxs = self.state['machine_state'][machine_idxs, 0]
        op_idxs = self.state['job_state'][job_idxs, 0]

        for job_id, op_id in zip(job_idxs, op_idxs):
            if np.random.rand() > self.failure_prob: # Operation is successful
                self.state['job_state'][job_id, 1] = 0
                if op_id == self.prb_instance.n_ops[job_id] - 1:
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
        """ Check if all the operations have been completed """
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
        """ Next time epoch is when the first time a machine finishes or when a pending operation is scheduled.
            In case of a tie finish time has priority.
            Returns:
                next_time_epoch (float): The next time epoch
                event_type (str): The type of event ['start', 'finish']
        """
        # 1) Find the next start time of the operations
        
        pending_ops = self.state['schedule_state'][:, :, 3] == 0
        pending_ops_start_times = self.state['schedule_state'][pending_ops, 0]
        
        # 2) Find the next finish time of the running operations
        
        non_idle_machines = self.state['machine_state'][:, 1] > 0
        non_idle_machines_durations = self.state['machine_state'][non_idle_machines, 1]
        if len(non_idle_machines_durations) == 0:
            next_finish_time = np.inf
        else:
            next_finish_time = np.min(non_idle_machines_durations) + self.state['current_time']

        if len(pending_ops_start_times) == 0:
            return next_finish_time, 'finish'
        next_start_time = np.min(pending_ops_start_times)
        
        # 3) In case of a tie finish time has priority
        if next_start_time == next_finish_time or next_finish_time < next_start_time:
            return next_finish_time, 'finish'
        elif next_start_time < next_finish_time:
            return next_start_time, 'start'
        else:
            raise ValueError('Invalid time epoch')
        
    def initialize_simulation(self, schedule) -> None:
        """ Initialize at time 0 for the given schedule, environment is:
            NOTE: The setup time is part of the operation duration in the schedule
            - Job state: 
                - All operation indices are set to 0
                - All jobs are set to idle
            - Machine state:
                - All machines are set to work on jobs in initial_setup.csv
            - Schedule state:
                - No jobs are scheduled
            - Current time: 0        
        """
        self.state['job_state'][:, 0] = 0
        self.state['job_state'][:, 1] = 0
        self.state['machine_state'][:, 0] = self.prb_instance.machines_initial_state
        self.state['schedule_state'] = schedule
        self.state['current_time'] = 0
        return None
        
    def get_state_space(self) -> gym.spaces.Dict:
        """ Return the state space of the environment """
        n_jobs = self.prb_instance.n_jobs
        max_ops = max(self.prb_instance.n_ops)
        n_machines = self.prb_instance.n_machines   
        state_space = gym.spaces.Dict({
            'job_state': gym.spaces.Box(
                low=np.tile(np.array([-1, 0], dtype=np.int32), (n_jobs, 1)),  # Repeat [-1, 0] for each job
                high=np.tile(np.array([max_ops-1, 1], dtype=np.int32), (n_jobs, 1)),  # Repeat [n_jobs, 1] for each job
                dtype=np.int32
            ),
            'machine_state': gym.spaces.Box(
                low=np.tile(np.array([-1, 0], dtype=np.int32), (n_machines, 1)),  # Repeat [-1, 0] for each machine
                high=np.tile(np.array([n_jobs, 10**6], dtype=np.int32), (n_machines, 1)),  # Repeat [n_jobs, inf] for each machine
                dtype=np.float32  # Use float32 since we include `np.inf`
            ),
            'schedule_state': gym.spaces.Box(
                low=np.tile(np.array([-1, -1, -1, -1], dtype=np.int32), (n_jobs, n_machines, 1)),  # Match shape (n_jobs, n_machines, 4)
                high=np.tile(np.array([10**6, 10**6, n_machines, 2], dtype=np.int32), (n_jobs, n_machines, 1)),  # Match shape
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
            'current_time': 0
        }
        return state

class GanttCharts(object):
    """ Class for rendering Gantt charts """

    def __init__(self, img_dir='images'):
        self.img_dir = f'images/{img_dir}'
        Path(self.img_dir).mkdir(parents=True, exist_ok=True)
        # Remove any existing images in the directory if any
        for file in os.listdir(self.img_dir):
            if file.endswith(".png"):
                os.remove(os.path.join(self.img_dir, file))

    def generate_gantt_chart(self, env, img_name: str = None, caption: str = None) -> None:
        """ Render the current state of the environment as a gannt chart:
            args:
                env: The environment (see envs/shopFloor.py)
                img_name: The name of the image file
        """
        path = f'{self.img_dir}/image_{img_name}.png'

        current_schedule = env.state['schedule_state']
        n_jobs = env.prb_instance.n_jobs
        n_machines = env.prb_instance.n_machines

        # Compute the longest completion time for x-axis 
        current_time = 0
        for job in current_schedule:
            for op in job:
                if np.all(op != [-1, -1, -1, -1]):
                    completion_time = op[0] + op[1]
                    current_time = max(current_time, completion_time)
        x_axis_limit = current_time + 10
        cmap = plt.cm.get_cmap('tab10')
        job_colors = [cmap(i) for i in range(n_jobs)]
        alpha_map = {0: 0.3, 1: 0.7, 2: 1.0}
        status_map = {0: 'P', 1: 'S', 2: 'C'} # Pending, Scheduled, Completed

        # Initialize the figure
        fig, gnt = plt.subplots(figsize=(12, 5))
        gnt.set_ylim(0, n_machines*10 + 10)
        gnt.set_xlim(0, x_axis_limit)
        gnt.set_xlabel('Time')
        gnt.set_ylabel('Machine')
        gnt.set_yticks([5 + 10*i + 4.5 for i in range(n_machines)])
        gnt.set_yticklabels([f'M: {i}' for i in range(n_machines)])
        gnt.grid(True)

        # Add the bars for each operation
        for job_idx, job in enumerate(current_schedule):
            for op_idx, (start, duration, machine_idx, status) in enumerate(job):
                if np.all([start, duration, machine_idx, status] != [-1, -1, -1, -1]):
                    gnt.broken_barh([(start, duration)], (5+10*machine_idx, 9), facecolors=job_colors[job_idx],
                                    alpha=alpha_map[status])
                    # Add a label at the center of each bar
                    text = f'o{op_idx}_{status_map[status]}'
                    #text = f'op{op_idx}_F' if is_finished else f'op{op_idx}_S'
                    if duration > 0:
                        gnt.text(start + duration/2, 5+10*machine_idx + 4.5,
                                text, color='black', fontsize=8, ha='center', va='center')    
                
        # Add a legend for the jobs
        job_legend_patches = [mpatches.Patch(color=job_colors[i], label=f'Job{i}') for i in range(n_jobs)]
        gnt.legend(handles=job_legend_patches, loc='center right', bbox_to_anchor=(1.10, 0.85))
        
        # Add a vertical line for the current time    
        current_timestamp = env.state["current_time"]  # Assuming the current timestamp is stored in the state
        gnt.axvline(current_timestamp, color='red', linestyle='--', label='Current Time')
        
        if caption:
            plt.title(caption + f'\n P: Pending, S: Scheduled, C: Completed')
        else:
            plt.title(f'P: Pending, S: Scheduled, C: Completed')



        # Save the figure
        plt.savefig(path)
    
    def construct_animation(self, failure_prob: float,
                            fps: int = 1, interval: int = 750, 
                            clear_img_folder: bool = False) -> None:
        """ Save an animation from a folder of images."""
        image_files = sorted([os.path.join(self.img_dir, file) for file in os.listdir(self.img_dir) if file.endswith(".png")],
                             key=lambda x: int(x.split('_')[-1].split('.')[0]))
        fig, ax = plt.subplots()
        plt.axis('off')  # Turn off axes

        # Placeholder for the image object
        img = plt.imshow(Image.open(image_files[0]))

        # Update function for animation
        def update(frame):
            img.set_data(Image.open(image_files[frame]))
            return [img]
        
        # Create the animation
        ani = FuncAnimation(fig, update, frames=len(image_files), interval=interval, blit=True)
        dir = 'animations'
        Path(dir).mkdir(parents=True, exist_ok=True)    
        output_file = f'{dir}/animation_failure_prob_{failure_prob}'
        ani.save(f'{output_file}.gif', writer="pillow", fps=fps)

        if clear_img_folder:
            for img in image_files:
                os.remove(img)
        return None