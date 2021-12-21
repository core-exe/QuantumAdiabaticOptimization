import os
import math
import argparse
import numpy as np
from numpy.core.numeric import outer
from black_box import FourierBlackBox
from bayes_opt import BayesianOptimization
from QuantumAnnealing.Three_SAT import get_3sat_problem
from QuantumAnnealing.GroverSearch import get_gs_problem
from tqdm import tqdm
import json

parser = argparse.ArgumentParser()
args = parser.parse_args()

n_samples = 100
data_dict = 'output/'

final_time = {4:32, 6:64, 8:128, 10:256}
sub_dir = {
    4:'qubit_4_t_32/',
    6:'qubit_6_t_64/',
    8:'qubit_8_t_128/',
    10:'qubit_10_t_256/',
}

black_box = {n: FourierBlackBox(get_3sat_problem,
                            n_qubit=n,
                            cutoff=6,
                            time_final=final_time[n],
                            time_step=1,
                            pround=(-0.1, 0.1)) for n in [4,6,8,10]}

acc = {
    i: {j: 0. for j in [4,6,8,10]} for i in [4,6,8,10] 
}

param_name = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6']
param_to_dict = lambda x: {param_name[i]: x[i] for i in range(len(param_name))}

params = {i: param_to_dict(np.load(data_dict+sub_dir[i]+"Multi-arm_Bandit_10.npy")[0][1:]) for i in [4,6,8,10]}

for i in [4,6,8,10]: # param
    for j in [4,6,8,10]: # n_qubit
        r = 0
        for _ in tqdm(range(n_samples), dynamic_ncols=True, desc='param {:02d} on problem {:02d}'.format(i, j)):
            r += black_box[j].black_box_reward(**params[i])
        acc[i][j] = r / n_samples
        print(acc[i][j])

with open(data_dict+'transfer.json', mode='w') as f:
    json.dump(acc, f)