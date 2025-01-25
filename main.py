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

    # Get the agent
    ## Add more agents here. Uncomment when implemented
    if args.priority_rule in ['edd', 'lpt', 'spt', 'wspt']:
        agent = StaticAgent(args.priority_rule)
    else:
        agent = DynamicAgent(args.priority_rule)

    env = ShopFloor(inst, gantt_plotter, agent, args.priority_rule, failure_prob=args.failure_prob)
    return env

def find_convergence(args):
    max_N = args.N  # Total number of iterations
    obj_values_estimate = np.zeros(max_N)  # Stores estimated objective values for each iteration
    plt.figure(figsize=(10, 5))
    for p in ['edd', 'lpt', 'spt', 'wspt', 'atcs', 'msf']:
        args.priority_rule = p
        for i in tqdm(range(1, max_N + 1), desc="Finding convergence of objective function estimate"):
            # Perform multiple simulations to estimate the objective value
            obj_values = np.zeros(i)  # Store objective function values for this iteration
            
            for j in tqdm(range(i), desc=f"Simulating scheduling for N={i}"):
                env = init_main(args)
                schedule = env.agent.get_schedule(env)
                obj_function = env.simulate_scheduling(schedule, plot_gantt=False)
                obj_values[j] = obj_function
            
            # Calculate the mean objective value for this iteration
            obj_values_estimate[i - 1] = np.mean(obj_values)
        # Save the values to a file
        np.save(f"objective_function_convergence_{args.priority_rule}.npy", obj_values_estimate)
        # Plot the results
        
        plt.plot(range(1, len(obj_values_estimate) + 1), obj_values_estimate, label=f"{args.priority_rule}")
    plt.xlabel("Number of Simulations (N)")
    plt.ylabel("Objective Function Estimate")
    plt.title(f"Convergence of Objective Function Estimate {args.priority_rule}")
    plt.legend()
    plt.savefig(f"objective_function_convergence_{args.priority_rule}.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--priority_rule", type=str, default="msf",                                 #qui metti le priorità
                        choices = ['edd', 'lpt', 'spt', 'wspt', 'atcs', 'msf'],
                        help="The priority rule to be used for the scheduling")
    parser.add_argument("--failure_prob", type=float, default=0.0)
    parser.add_argument("--clear_imgs", action='store_true', help="Clear the gantt charts when creating the animation")
    parser.add_argument("--N", type=int, default=1, help="Number of iterations for the simulation")
    args = parser.parse_args()

    # Find the convergence of the objective function estimate
    find_convergence(args)

    # 1) Initialize the environment, agent and compute the initial schedule
    # env = init_main(args)
    # schedule = env.agent.get_schedule(env)
    # obj_function = env.simulate_scheduling(schedule, plot_gantt=args.N == 1)
    # print(f"Objective function value: {obj_function}")
    # env.gantt_plotter.construct_animation(failure_prob=env.failure_prob, priority_rule=env.priority_rule,
    #                                        interval=1400, clear_img_folder=args.clear_imgs)




    # # 2) Simulate the scheduling
    # obj_values_estimate = np.zeros(args.N)
    # for i in tqdm(range(args.N), desc="Simulating scheduling"):
    #     obj_values = np.zeros(i)
    #     for idx in range(i):
    #         env = init_main(args)
    #         schedule = env.agent.get_schedule(env)
    #         obj_function = env.simulate_scheduling(schedule, plot_gantt=args.N == 1)
    #         obj_values[idx] = obj_function
    #     obj_values_estimate[i] = np.mean(obj_values)
    # # Plot objective function values as a function of the iteration
    # plt.plot(obj_values)
    # plt.xlabel('Iteration')
    # plt.ylabel('Objective function value')
    # plt.savefig('objective_function.png')

    # # 3) Construct the animation
    # if args.N == 1:
    #     env.gantt_plotter.construct_animation(failure_prob=env.failure_prob, interval=1400, clear_img_folder=args.clear_imgs)
