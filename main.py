#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import logging
from envs import *
from agents import *
from data_interfaces import read_jit_jss_setup_instances
import argparse

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
    parser.add_argument("--priority_rule", type=str, default="edd", 
                        choices = ['edd', 'lpt', 'spt', 'wspt', 'atcs', 'msf'],
                        help="The priority rule to be used for the scheduling")
    parser.add_argument("--failure_prob", type=float, default=0.05)
    parser.add_argument("--clear_imgs", action='store_true', help="Clear the gantt charts when creating the animation")
    args = parser.parse_args()

    # 1) Initialize the environment, agent and compute the initial schedule
    env = init_main(args)
    schedule = env.agent.get_schedule(env)

    # 2) Simulate the scheduling
    obj_function = env.simulate_scheduling(schedule, plot_gantt=True)    
    print(f"Objective function value: {obj_function}")

    # 3) Construct the animation
    env.gantt_plotter.construct_animation(failure_prob=env.failure_prob, interval=1400, clear_img_folder=args.clear_imgs)

