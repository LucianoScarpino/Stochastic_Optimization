#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import logging
from envs import *
from agents import *
from data_interfaces import read_jit_jss_setup_instances
if __name__ == "__main__":
    log_name = os.path.join(
        '.', 'logs',
        f"{os.path.basename(__file__)[:-3]}.log"
    )
    logging.basicConfig(
        filename=log_name,
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO, datefmt="%H:%M:%S",
        filemode='w'
    )
    path_file = os.path.join(
        '.', 'data',
        'I-5x10-equal-loose-0'
    )
    # read static data
    inst = read_jit_jss_setup_instances(path_file)
    # create dynamic environment
    gantt_plotter = GanttCharts(img_dir='gantt_images')

    env = ShopFloor(inst, gantt_plotter)
    # create agent
    agent = EddAgent()
    # compute schedule
    schedule = agent.get_schedule(env)
    # render gannt chart
    env.render_gantt_chart()

    #simulate schedule
    obj_function = env.simulate_scheduling(schedule, plot_gantt=True)    
    print(f"Objective function value: {obj_function}")
    gantt_plotter.construct_animation(failure_prob=env.failure_prob)

