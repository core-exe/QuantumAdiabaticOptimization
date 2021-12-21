import os
import math
import argparse
import numpy as np
from black_box import FourierBlackBox
from bayes_opt import BayesianOptimization
from QuantumAnnealing.Three_SAT import get_3sat_problem
from QuantumAnnealing.GroverSearch import get_gs_problem
from tqdm import tqdm

def get_split(param_list):
    value_list = sorted(param['target'] for param in param_list)
    return value_list[len(value_list) // 2]

def save_results(param_list, path):
    param_array = []
    for param in param_list:
        param_array.append([param['target']] + list(param['params'].values()))
    param_array = np.array(param_array)
    np.save(path, param_array)

parser = argparse.ArgumentParser()
parser.add_argument('--n_qubit', type=int, default=8)
parser.add_argument('--cutoff', type=int, default=6)
parser.add_argument('--time_final', type=float, default=62.2)
parser.add_argument('--time_step', type=float, default=1)
parser.add_argument('--pround', type=float, default=0.1)
parser.add_argument('--n_sample', type=int, default=10)
parser.add_argument('--n_point', type=int, default=1024)
parser.add_argument('--output_dir', type=str, default='output')
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

black_box_class = FourierBlackBox(get_3sat_problem,
                                  n_qubit=args.n_qubit,
                                  cutoff=args.cutoff,
                                  time_final=args.time_final,
                                  time_step=args.time_step,
                                  pround=(-args.pround, args.pround),
                                  num_sample=args.n_sample)

print('Gaussian process started.')

optimizer = BayesianOptimization(black_box_class.black_box_reward,
                                 black_box_class.prounds,
                                 verbose=2)

optimizer.probe([0] * black_box_class.cutoff)
optimizer.maximize(init_points=0, n_iter=args.n_point-1)

print('Gaussian process completed.')

param_list = optimizer.res.copy()
save_results(param_list, os.path.join(args.output_dir, 'Gaussian_Process.npy'))

for i in range(len(param_list)):
    param_list[i]['id'] = i

num_iter = int(math.ceil(math.log2(len(param_list))))
per_round_budget = int(math.ceil(len(param_list) / num_iter))

for i in range(1, num_iter + 1):
    split_num = get_split(param_list)
    n_hyperband_sample = int(math.ceil(2 * per_round_budget / len(param_list)))
    new_param_list = []
    pbar = tqdm(total=n_hyperband_sample*len(param_list)//2, dynamic_ncols=True)
    pbar.set_description("Step {:02d} | Threshold = {:.4f}".format(i, split_num))
    for param in param_list:
        if param['target'] >= split_num:
            reward = 0
            for _ in range(n_hyperband_sample):
                reward += black_box_class.black_box_reward(**param['params'])
                pbar.update()
            reward /= n_hyperband_sample
            param['target'] = reward
            new_param_list.append(param)
    pbar.close()
    param_list = new_param_list
    save_results(param_list, os.path.join(args.output_dir, 'Multi-arm_Bandit_%d') % i)

print('Multi-arm bandit completed.')
print(param_list[0])
