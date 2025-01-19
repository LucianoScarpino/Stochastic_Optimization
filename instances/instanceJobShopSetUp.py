# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from .instance import Instance


class InstanceJobShopSetUp(Instance):
    """
    Class to manage the instance of the job shop scheduling problem with setup times.
        attributes:
            - n_jobs (int): number of jobs (5)
            - n_machines (int): number of machines (10)
            - machines_initial_state (list[int]): initial state of the machines (initial_setup.csv)
            - df_jobs (pd.DataFrame): dataframe with the jobs information (jobs.csv)
            - df_setup_job (pd.DataFrame): dataframe with the setup information (setup.csv)
            - lst_job (list[list[dict]]): list of jobs, each job is a list of 
                        dictionaries with the machine and the processing time (operations.csv)
            - op_from_j_m (np.array): contains the number of the operation given the job and the machine, -1 if machine is not used for the job
            - jobs_on_machine (list[list[int]]): contains all the jobs to be done on a machine
            - n_ops (list[int]): number of operations for each job
            - jobs (dict): contains the jobs information 
            - operations_forest (nx.DiGraph): graph with the operations 
            - df_operations (pd.DataFrame): dataframe with the operations information
            - df_setup (pd.DataFrame): dataframe with the setup information (setup.csv)
            - lst_operations (list): list of operations each operation is a tuple (op, job)
            - eligible_machines (dict): eligible machines for each operation
            - starting_ops (list): starting operations

        methods:
            - get_setup(m, j0, j1): get the setup time between two jobs on a machine
            - plot_operations_forest(): plot the operations forest
    """

    def __init__(self, n_jobs, n_machines, machines_initial_state, df_jobs, df_setup_job, lst_job):
        super(InstanceJobShopSetUp, self).__init__(n_jobs, n_machines)
        self.machines_initial_state = machines_initial_state
        self.lst_machines = range(n_machines)
        self.df_jobs = df_jobs
        self.df_setup_job = df_setup_job
        self.lst_job = lst_job
        # contains the number of the operation given the job and the machine
        self.op_from_j_m = - np.ones(shape=(
            n_jobs, n_machines
        ), dtype=int)
        # contains all the jobs to be done on a machine
        self.jobs_on_machine = [[] for _ in range(self.n_machines)]
        for idx_job, job in enumerate(self.lst_job):
            for idx_op, op in enumerate(job):
                self.op_from_j_m[idx_job, op['machine']] = idx_op
                self.jobs_on_machine[op['machine']].append(idx_job)
        # n_operations for each job:
        self.n_ops = [ max(self.op_from_j_m[idx_job,:]) + 1 for idx_job in range(n_jobs) ]
        # need for the gantt plot
        self.jobs = {}
        for _, row in self.df_jobs.iterrows():
            self.jobs[int(row['id_job'])] = {
                'item_name': row['id_job'],
                'due_date': row['due_date'],
                'release_date': row['release_date'],
                'earliness_penalty': row['earliness_penalty'],
                'tardiness_penalty': row['tardiness_penalty'],
                'flow_time_penalty': row['flow_time_penalty'],
            }
        operations = []
        self.operations_forest = nx.DiGraph()
        for idx_job, job in enumerate(lst_job):
            for idx_op, op in enumerate(job):
                earliest_starting = self.df_jobs.iloc[idx_job].due_date - sum(
                    [job[i]['processing_time'] for i in range(idx_op, len(job))]
                )
                operations.append(
                    {
                        'op': (idx_op, idx_job),
                        'order': idx_job,
                        'id_order': f"ord{idx_job}",
                        'machines': op['machine'],
                        'duration_h': op['processing_time'],
                        'job_due_date': df_jobs.iloc[idx_job]['due_date'],
                        'importance': df_jobs.iloc[idx_job]['earliness_penalty'] + df_jobs.iloc[idx_job]['tardiness_penalty'],
                        'earliest_starting': earliest_starting
                    }
                )

                self.operations_forest.add_node(
                    (idx_op, idx_job),
                    pos = (idx_op, idx_job)
                )
                if idx_op > 0:
                    self.operations_forest.add_edge(
                        (idx_op - 1, idx_job),
                        (idx_op, idx_job),
                    )
        self.df_operations = pd.DataFrame.from_dict(operations)
        # setup operations
        setups_tmp = []
        for _, row in self.df_setup_job.iterrows():
            setups_tmp.append(
                {
                    "machine": row.machine,
                    "op1": (self.op_from_j_m[row.id_job0, row.machine], row.id_job0),
                    "op2": (self.op_from_j_m[row.id_job1, row.machine], row.id_job1),
                    "time_h": row.time,
                }
            )
        self.df_setup = pd.DataFrame.from_dict(setups_tmp)
        # COMUNE A TUTTE LE INSTANCE:
        self.lst_operations = list(self.df_operations.op.unique())
        # Eligible machine: key op -> list of machines
        self.eligible_machines = {}
        for op in self.lst_operations:
            self.eligible_machines[op] = list(self.df_operations[self.df_operations.op == op].machines)
        
        # needed for the list scheduler
        self.starting_ops = [x for x in self.operations_forest.nodes() if len(list(self.operations_forest.predecessors(x))) == 0]# starting operations

    def get_setup(self, m, j0, j1):
        ans = self.df_setup_job[
            (self.df_setup_job.machine==m) & (self.df_setup_job.id_job0==j0) & (self.df_setup_job.id_job1==j1)
        ]
        if len(ans) == 1:
            return ans.iloc[0]['time']
        else:
            return 0

    def plot_operations_forest(self):
        nx.draw(
            self.operations_forest,
            pos=nx.get_node_attributes(
                self.operations_forest,'pos'
            )
        )
        plt.show()