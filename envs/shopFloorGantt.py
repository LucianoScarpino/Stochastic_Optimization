import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from PIL import Image
import os
from pathlib import Path
import numpy as np

class GanttCharts(object):
    """ Class for rendering Gantt charts of the shop floor scheduling environment.
        args:
            img_dir: The directory to save the images
        methods:
            generate_gantt_chart: Generate a gantt chart of the current state of the environment
            construct_animation: Construct an animation from a folder of images
    """
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
                caption: Extra information to add to the title of the image
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

        # Pending, Scheduled, Completed status
        alpha_map = {0: 0.3, 1: 0.7, 2: 1.0}    
        status_map = {0: 'P', 1: 'S', 2: 'C'} 

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
                    if duration > 0:
                        gnt.text(start + duration/2, 5+10*machine_idx + 4.5,
                                text, color='black', fontsize=8, ha='center', va='center')    
                
        # Add a legend for the jobs
        job_legend_patches = [mpatches.Patch(color=job_colors[i], label=f'Job{i}') for i in range(n_jobs)]
        gnt.legend(handles=job_legend_patches, loc='center right', bbox_to_anchor=(1.10, 0.85))
        
        # Add a vertical line for the current time    
        current_timestamp = env.state["current_time"]  # Assuming the current timestamp is stored in the state
        gnt.axvline(current_timestamp, color='red', linestyle='--', label='Current Time')
        
        if caption is not None:
            plt.title(caption + f'\n P: Pending, S: Scheduled, C: Completed')
        else:
            plt.title(f'P: Pending, S: Scheduled, C: Completed')

        # Save the figure
        plt.savefig(path)
    
    def construct_animation(self, failure_prob: float, priority_rule: str,
                            fps: int = 1, interval: int = 1400, 
                            clear_img_folder: bool = True) -> None:
        """ Save an animation from a folder of images.
            args:
                failure_prob: The probability of failure
                priority_rule: The priority rule used to schedule the jobs
                fps: The frames per second of the animation
                interval: The interval between frames
                clear_img_folder: Whether to clear the images in the folder after creating the animation
        """
        if not clear_img_folder:
            print('Warning: The images will not be cleared from the folder after creating the animation.')

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
        output_file = f'{dir}/{priority_rule}_failure_{failure_prob}'
        ani.save(f'{output_file}.gif', writer="pillow", fps=fps)

        if clear_img_folder:
            for img in image_files:
                os.remove(img)
        return None