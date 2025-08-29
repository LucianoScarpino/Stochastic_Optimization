#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
from envs import *
from agents import *
from data_interfaces import *
import argparse
import numpy as np
from tqdm import tqdm

def mean_std_ci(obj_values: np.ndarray, alpha: float = 0.05) -> tuple[float, float, tuple[float, float]]:
    """
    Compute descriptive statistics for the objective values obtained across scenarios.

    Args:
        obj_values (np.ndarray): Array of objective function values, one per scenario.
        alpha (float): Significance level for the confidence interval (default 0.05).

    Returns:
        mean (float): Mean objective value.
        std (float): Standard deviation of objective values.
        ci (tuple[float,float]): Lower and upper bounds of the (1-alpha) confidence interval,
                                 using normal approximation (z≈1.96).
    """
    obj_values = np.asarray(obj_values, dtype=float)
    n = len(obj_values)
    m = float(np.mean(obj_values)) if n else float('nan')
    s = float(np.std(obj_values, ddof=1)) if n > 1 else 0.0
    if n <= 1:
        return m, s, (m, m)
    z = 1.96
    half = z * s / np.sqrt(n)
    return m, s, (m - half, m + half)

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

def run(priority_rule=None,failure_prob=None,scenario_file=None,test=False,printed=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--priority_rule", type=str, default="edd",                                 
                        choices = ['edd', 'lpt', 'spt', 'wspt', 'atcs', 'msf'],
                        help="The priority rule to be used for the scheduling")
    parser.add_argument("--failure_prob", type=float, default=0.1)
    parser.add_argument("--n", type=int, default=1, help="Number of iterations for the simulation")
    parser.add_argument("--scenario_file", type=str, default=None,
                        help="Path to a JSON file containing scenario seeds (for CRN). If not provided, a new set is generated." \
                             "use ./data/scenarios/name_scenario.json")
    parser.add_argument("--base_seed", type=int, default=12345,
                        help="Base seed used to generate scenario seeds when --scenario_file is not provided.")
    args = parser.parse_args()

    if priority_rule != None and test == True:
        args.priority_rule = priority_rule
    if failure_prob != None and test == True:
        args.failure_prob = failure_prob
    if scenario_file != None and test == True:
        args.scenario_file = scenario_file

    if args.scenario_file:
        abs_path = os.path.abspath(args.scenario_file)
        if not printed:
            print('-'*50)
            print(f"[DEBUG] scenario_file provided: {args.scenario_file} (abs: {abs_path}), exists={os.path.isfile(args.scenario_file)}")
            print('-'*50)
    else:
        if not printed:
            print('-'*50)
            print("[DEBUG] no scenario_file provided; will use generator")
            print('-'*50)

    if args.scenario_file and os.path.isfile(args.scenario_file):
        if not printed:
            print('-'*50)
            print("[DEBUG] Loading scenarios from file...")
            print('-'*50)
        scen_set = ScenarioGenerator.load(args.scenario_file)
    else:
        if not printed:
            print('-'*50)
            print(f"[DEBUG] Generating scenarios with n={args.n}, base_seed={args.base_seed}...")
            print('-'*50)
        scen_set = ScenarioGenerator(n=args.n, base_seed=args.base_seed).generate()

        # Save scenarios if a path was provided (enables reuse across rules for CRN)
        if args.scenario_file:
            if not printed:
                print('-'*50)
                print(f"[DEBUG] Saving generated scenarios to {args.scenario_file}")
                print('-'*50)
            ScenarioGenerator.save(args.scenario_file, scen_set)

    # Ensure n matches the actual number of scenarios loaded/generated
    if args.n != len(scen_set.seeds):
        if not printed:
            print('-'*50)
            print(f"[INFO] Overriding --n: using n={len(scen_set.seeds)-1} from scenario file lenght instead of n={args.n}")
            print('-'*50)
        args.n = len(scen_set.seeds)
        

    # 1) Initialize the environment
    # 2) Simulate the scheduling
    obj_values = np.zeros(args.n)
    for i, seed in enumerate(tqdm(scen_set.seeds, desc="Simulating scheduling")):
        np.random.seed(seed)  # set scenario
        env = init_main(args)

        # On the first scenario of this execution, clear previous frames
        if i == 0:
            img_dir = getattr(env.gantt_plotter, 'img_dir', None)
            if img_dir and os.path.isdir(img_dir):
                for f in os.listdir(img_dir):
                    if f.endswith('.png'):
                        try:
                            os.remove(os.path.join(img_dir, f))
                        except OSError:
                            pass

        schedule = env.agent.get_schedule(env)
        last_scenario = (i == args.n - 1)
        obj_function = env.simulate_scheduling(schedule, plot_gantt=last_scenario)
        obj_values[i] = obj_function

    # Generate metrics
    mean_obj,std_obj,I_obj = mean_std_ci(obj_values)
    print('-'*50)
    print(f'Priority rule: {args.priority_rule}')
    print(f'Failure probability: {args.failure_prob}')
    print(f"Estimated objective function value: {mean_obj}")
    print('-'*50)
    
    # 3) Construct the animation
    if not test:
        env.gantt_plotter.construct_animation(args.failure_prob, args.priority_rule)
    else:
        return mean_obj,std_obj,I_obj

if __name__ == "__main__":
    run()