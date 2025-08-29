#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
from envs import *
from agents import *
from data_interfaces import *
import argparse
import numpy as np
from tqdm import tqdm

def mean_std_ci(values: np.ndarray, alpha: float = 0.05, weights: np.ndarray = None) -> tuple[float, float, tuple[float, float]]:
    """
    Compute (weighted) mean, std and (1-alpha) CI (normal approx).
    - If weights=None -> unweighted (original behavior).
    - If weights given  -> weighted with n_eff-based CI.
    """
    x = np.asarray(values, dtype=float)
    if weights is None:
        n = len(x)
        m = float(np.mean(x)) if n else float('nan')
        s = float(np.std(x, ddof=1)) if n > 1 else 0.0
        if n <= 1:
            return m, s, (m, m)
        z = 1.96
        half = z * s / np.sqrt(n)
        return m, s, (m - half, m + half)

    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) != len(x):
        raise ValueError("weights must be a 1D array with same length as values")
    sum_w = float(w.sum())
    if sum_w <= 0:
        # fallback: treat as unweighted
        return mean_std_ci(x, alpha=alpha, weights=None)

    w = w / sum_w
    m = float(np.sum(w * x))
    # weighted variance around weighted mean
    var = float(np.sum(w * (x - m) ** 2))
    # effective sample size
    n_eff = 1.0 / float(np.sum(w ** 2))
    # If a weight is dominant between others n_eff == 1, no variability
    if n_eff <= 1:
        return m, 0.0, (m, m)

    # small-sample correction (Bessel-like with n_eff)
    var *= n_eff / (n_eff - 1.0)
    s = float(np.sqrt(var))
    z = 1.96  # normal approx
    half = z * s / np.sqrt(n_eff)
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

def run(priority_rule=None,failure_prob=None,scenario_file=None,test=False,printed=False,
        reduction=False,k=None):
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
    parser.add_argument("--use_reduction", action="store_true",
                    help="If set, apply scenario reduction (k-medoids) before simulation.")
    parser.add_argument("--k_scenarios", type=int, default=15,
                    help="Number of scenarios to keep after reduction (k<=n). Ignored if --use_reduction is not set.")
    parser.add_argument("--embed_length", type=int, default=1000,
                    help="Length of the Bernoulli stream used to build the scenario embedding.")
    args = parser.parse_args()

    if priority_rule != None and test == True:
        args.priority_rule = priority_rule
    if failure_prob != None and test == True:
        args.failure_prob = failure_prob
    if scenario_file != None and test == True:
        args.scenario_file = scenario_file
    if reduction != None and test == True:
        args.use_reduction = reduction
    if k != None and test == True:
        args.k_scenarios = k

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

    if args.use_reduction and args.k_scenarios < len(scen_set.seeds):
        # Costruisci la matrice X (m x d) con gli embedding
        Xi = np.vstack([
            ScenarioReducer.make_scenario_embedding(seed=s,
                                    failure_probability=args.failure_prob,
                                    stream_length=args.embed_length)
            for s in scen_set.seeds
        ])
        # Riduci con k-medoids
        reducer = ScenarioReducer(k=args.k_scenarios, rng_seed=args.base_seed)
        keep_idx, weights, _ = reducer.reduce(Xi)
        selected_seeds = [scen_set.seeds[i] for i in keep_idx]
    else:
        # NESSUNA riduzione: simulo tutti gli scenari, pesi uniformi
        selected_seeds = list(scen_set.seeds)

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
    obj_values = np.zeros(len(selected_seeds))
    for i, seed in enumerate(tqdm(selected_seeds, desc="Simulating scheduling")):
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
        last_scenario = (i == len(selected_seeds) - 1)
        obj_function = env.simulate_scheduling(schedule, plot_gantt=last_scenario)
        obj_values[i] = obj_function

    # Generate metrics
    if args.use_reduction == True:
        mean_obj,std_obj,I_obj = mean_std_ci(obj_values,weights=weights)
    else:
        mean_obj,std_obj,I_obj = mean_std_ci(obj_values,weights=None)
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