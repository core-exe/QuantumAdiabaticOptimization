import math
import argparse
from black_box import FourierBlackBox
from bayes_opt import BayesianOptimization
from QuantumAnnealing.Three_SAT import get_3sat_problem, get_default_driver_3sat
import numpy as np
from tqdm import tqdm

n_qubits = 6
n_samples = 100

t_start = 64
t_end = 64
t_step_search = 2

t_list = []
reward_mean_list = []
reward_std_list = []

iterator = tqdm(range(t_start, t_end+t_step_search, t_step_search), dynamic_ncols=True)

for t in iterator:
    reward = []
    for _ in range(n_samples):
        problem = get_3sat_problem(n_qubits)
        driver = get_default_driver_3sat(problem, t, 0.2)
        reward.append(1-problem.loss(driver.simulate()[0]))
    out_string = "T_end = {:04d} | reward = {:.5f} +- {:.5f}".format(t, np.mean(reward), np.std(reward))
    print(out_string)
    iterator.set_description(out_string)
    t_list.append(t)
    reward_mean_list.append(np.mean(reward))
    reward_std_list.append(np.std(reward))

    
