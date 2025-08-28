#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
from envs import *
from agents import *
from data_interfaces import *
import argparse
import numpy as np
from tqdm import tqdm

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
    parser.add_argument("--scenario_file", type=str, default=None,
                        help="Path to a JSON file containing scenario seeds (for CRN). If not provided, a new set is generated.")
    parser.add_argument("--base_seed", type=int, default=12345,
                        help="Base seed used to generate scenario seeds when --scenario_file is not provided.")
    args = parser.parse_args()

    if args.scenario_file and os.path.isfile(args.scenario_file):
        scen_set = ScenarioGenerator.load(args.scenario_file)
    else:
        scen_set = ScenarioGenerator(n=args.n, base_seed=args.base_seed).generate()
        # Save scenarios if a path was provided (enables reuse across rules for CRN)
        if args.scenario_file:
            ScenarioGenerator.save(args.scenario_file, scen_set)

    # 1) Initialize the environment
    # 2) Simulate the scheduling
    obj_values = np.zeros(args.n)
    for i in tqdm(range(args.n), desc="Simulating scheduling"):         #tqdm for bar visualization
        env = init_main(args)
        schedule = env.agent.get_schedule(env)
        obj_function = env.simulate_scheduling(schedule, plot_gantt=(i == args.n-1)) # Plot the last simulation
        obj_values[i] = obj_function
    print(f"Estimated objective function value: {np.mean(obj_values)}")

    # 3) Construct the animation
    env.gantt_plotter.construct_animation(args.failure_prob, args.priority_rule)

