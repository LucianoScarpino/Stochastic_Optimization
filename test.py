import os
import pandas as pd
from main import *

from itertools import product

def generate_combinations(rules, failure_probs):
    """Return all (rule, failure_prob) combinations."""
    return [(r, p) for r, p in product(rules, failure_probs)]

def store_results(results,reduction=False,k=None,e_length=None):
    rows = []
    for (rule, p_fail), (mean, std, ci) in results.items():
        rows.append([rule, p_fail, mean, std,ci])
    
    df = pd.DataFrame(rows, columns=['rule','failure_prob','obj_mean','obj_std','obj_95_I'])
    
    results_dir = './results/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    existing_files = [f for f in os.listdir(results_dir) if f.startswith('results_') and f.endswith('.csv')]
    numbers = []
    for f in existing_files:
        parts = f.split('_')
        if len(parts) >= 2:
            num_part = parts[1]
            num_str = ''.join(filter(str.isdigit, num_part))
            if num_str.isdigit():
                numbers.append(int(num_str))
    next_num = max(numbers) + 1 if numbers else 1
    
    suffix = '-' + str(k) + '_' +str(e_length) + '_reducted.csv' if reduction else '_no_reducted.csv'
    filename = f'results_{next_num}{suffix}'
    filepath = os.path.join(results_dir, filename)
    
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")
    
    return df

if __name__ == "__main__":
    # Set test parameters (baseline)
    scenario_file = "./data/scenarios/test_scenario_50.json"
    rules = ['edd', 'lpt', 'spt', 'wspt', 'atcs', 'msf']
    failure_probs = [0.1,0.2,0.3]

    # Set reduction parameters
    scenario_reduction = True
    k = 15
    embeed_length = 1000
    
    # Create combinations for testing
    combinations = generate_combinations(rules,failure_probs)
    print('-'*50)
    print(f'Number of combinations to test: {len(combinations)}')
    print('-'*50)

    # Testing
    printed_info = False # Avoid multiple prints info
    tested = {}
    for comb in combinations:
        if scenario_reduction == False:
            m_obj,std_obj,I_obj = run(comb[0],comb[1],scenario_file,test=True,printed=printed_info)
        else:
            m_obj,std_obj,I_obj = run(comb[0],comb[1],scenario_file,test=True,printed=printed_info,
                                      reduction=scenario_reduction,k=k,e_length=embeed_length)
        printed_info = True
        

        # Store metrics
        tested[comb] = [m_obj,std_obj,I_obj]
    
    # Save results of test
    match scenario_reduction:
        case False:
            store_results(tested,reduction = False) 
        case True:
            store_results(tested,reduction = True,k=k,e_length=embeed_length) 
