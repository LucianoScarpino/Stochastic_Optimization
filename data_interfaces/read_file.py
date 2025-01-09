import os
import json
import pandas as pd
from instances import *

def read_jit_jss_setup_instances(path_file: str) -> InstanceJobShopSetUp:
    # READ GENERAL INFO
    fp_setting = open(
        os.path.join(
            path_file,  
            "settings.json"
        ), 'r'
    )
    settings = json.load(fp_setting)
    fp_setting.close()
    n_jobs = settings['n_jobs']
    n_machines = settings['n_machines']
    # READ JOBS INFO
    df_jobs = pd.read_csv(
        os.path.join(
            path_file,  
            "jobs.csv"
        )
    )
    # READ SETUP INFO
    df_setup = pd.read_csv(
        os.path.join(
            path_file,  
            "setup.csv"
        )
    )
    fp_operations = open(
        os.path.join(
            path_file,  
            "operations.csv"
        ), 'r'
    )
    rows = fp_operations.readlines()
    lst_job = [ [] for _ in range(n_jobs)]
    # for each line
    for idx_job, row in enumerate(rows):
        row_split = row.strip().split(',')
        for i in range(0, len(row_split), 2):
            lst_job[idx_job].append(
                {    
                    "machine": int(row_split[i]),
                    "processing_time": int(row_split[i + 1])
                }
            )
    fp_operations.close()
    
    fp_initial_setup = open(
        os.path.join(
            path_file,  
            "initial_setup.csv"
        ), 'r'
    )
    machines_initial_state = []
    rows = fp_initial_setup.readlines()
    for i, row in enumerate(rows):
        if i == 0:
            continue
        else:
            machine_state, state = row.strip().split(',')
            machines_initial_state.append(int(state))

    fp_initial_setup.close()
    return InstanceJobShopSetUp(n_jobs, n_machines, machines_initial_state, df_jobs, df_setup, lst_job)
