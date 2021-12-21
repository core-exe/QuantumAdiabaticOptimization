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

parser = argparse.ArgumentParser()
parser.add_argument('--n_qubit', type=int, default=8)
parser.add_argument('--data_dict', type=str)
parser.add_argument('--samples', type=int, default=2048)
args = parser.parse_args()

final_time = {4:32, 6:64, 8:128, 10:256}

black_box_class = FourierBlackBox(get_3sat_problem,
                                  n_qubit=args.n_qubit,
                                  cutoff=6,
                                  time_final=final_time[args.n_qubit],
                                  time_step=1,
                                  pround=(-0.1, 0.1),
                                  num_sample=1)

linear_acc = np.zeros((args.samples, ))
linear_param = np.zeros((6, ))
opt_acc = np.zeros((args.samples, ))
opt_param = np.load(args.data_dict + 'Multi-arm_Bandit_10.npy')[0][1:]

param_name = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6']
param_to_dict = lambda x: {param_name[i]: x[i] for i in range(len(param_name))}

for i in tqdm(range(args.samples), dynamic_ncols=True):
    linear_acc[i] = black_box_class.black_box_reward(**param_to_dict(linear_param))
    opt_acc[i] = black_box_class.black_box_reward(**param_to_dict(opt_param))

np.save(args.data_dict + 'Linear_distribution.npy', linear_acc)
np.save(args.data_dict + 'Opt_distribution.npy', opt_acc)