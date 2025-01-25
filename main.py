#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import logging
from envs import *
from agents import *
from data_interfaces import read_jit_jss_setup_instances
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp    

## Add more agents here. Uncomment when implemented
priority_to_agent = {
    'edd': EddAgent()}
    # 'lpt': LptAgent,
    # 'spt': SptAgent,
    # 'wspt': WsptAgent,
    # 'atcs': AtcsAgent,
    # 'msf': MsfAgent}

def init_main(args: argparse.Namespace) -> ShopFloor:
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

    agent = priority_to_agent[args.priority_rule]

    env = ShopFloor(inst, gantt_plotter, agent, args.priority_rule, failure_prob=args.failure_prob)
    return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--priority_rule", type=str, default="edd",                                 #qui metti le priorità
                        choices = ['edd', 'lpt', 'spt', 'wspt', 'atcs', 'msf'],
                        help="The priority rule to be used for the scheduling")
    parser.add_argument("--failure_prob", type=float, default=0.0)
    parser.add_argument("--clear_imgs", action='store_true', help="Clear the gantt charts when creating the animation")
    parser.add_argument("--N", type=int, default=100, help="Number of iterations for the simulation")
    args = parser.parse_args()

    # 1) Initialize the environment, agent and compute the initial schedule

    # 2) Simulate the scheduling
    obj_values_estimate = np.zeros(args.N)
    for i in tqdm(range(args.N), desc="Simulating scheduling"):
        obj_values = np.zeros(i)
        for idx in range(i):
            env = init_main(args)
            schedule = env.agent.get_schedule(env)
            obj_function = env.simulate_scheduling(schedule, plot_gantt=args.N == 1)
            obj_values[idx] = obj_function
        obj_values_estimate[i] = np.mean(obj_values)
    # Plot objective function values as a function of the iteration
    plt.plot(obj_values)
    plt.xlabel('Iteration')
    plt.ylabel('Objective function value')
    plt.savefig('objective_function.png')

    # 3) Construct the animation
    if args.N == 1:
        env.gantt_plotter.construct_animation(failure_prob=env.failure_prob, interval=1400, clear_img_folder=args.clear_imgs)
