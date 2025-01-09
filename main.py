#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import logging
from envs import *
from agents import *
from data_interfaces import read_jit_jss_setup_instances


if __name__ == "__main__":
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
    env = ShopFloor(inst)
    # create agent
    agent = EddAgent()
    # compute schedule
    schedule = agent.get_schedule(env)
    # simulate schedule
    obj_function = env.simulate_scheduling(schedule)

