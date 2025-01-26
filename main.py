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
import json

def init_main(args: argparse.Namespace) -> ShopFloorSimulation:
    path_file = os.path.join(
        '.', 'data',
        'I-5x10-equal-loose-0')
    
    # Read static data
    inst = read_jit_jss_setup_instances(path_file)

    # Initialize the agent
    if args.priority_rule in ['edd', 'lpt', 'spt', 'wspt']:
        agent = StaticAgent(args.priority_rule)
    else:
        agent = DynamicAgent(args.priority_rule)

    # Initialize the environment
    env = ShopFloorSimulation(inst, agent, failure_prob=args.failure_prob)
    return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--priority_rule", type=str, default="edd",                                 
                        choices = ['edd', 'lpt', 'spt', 'wspt', 'atcs', 'msf'],
                        help="The priority rule to be used for the scheduling")
    parser.add_argument("--failure_prob", type=float, default=0.1)
    parser.add_argument("--n", type=int, default=1, help="Number of iterations for the simulation")
    args = parser.parse_args()

    # 1) Initialize the environment

    # 2) Simulate the scheduling
    obj_values = np.zeros(args.n)
    for i in tqdm(range(args.n), desc="Simulating scheduling"):
        env = init_main(args)
        schedule = env.agent.get_schedule(env)
        obj_function = env.simulate_scheduling(schedule, plot_gantt=(i == args.n-1)) # Plot the last simulation
        obj_values[i] = obj_function
    print(f"Estimated objective function value: {np.mean(obj_values)}")

    # 3) Construct the animation
    env.gantt_plotter.construct_animation(args.failure_prob, args.priority_rule)

