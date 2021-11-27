import math
import argparse
from black_box import FourierBlackBox
from bayes_opt import BayesianOptimization
from QuantumAnnealing.Three_SAT import get_3sat_problem
from QuantumAnnealing.GroverSearch import get_gs_problem

def get_split(param_list):
    value_list = sorted(param['target'] for param in param_list)
    return value_list[len(value_list) // 2]

parser = argparse.ArgumentParser()
parser.add_argument('--n_qubit', type=int, default=4)
parser.add_argument('--cutoff', type=int, default=6)
parser.add_argument('--time_final', type=float, default=62.2)
parser.add_argument('--time_step', type=float, default=0.2)
parser.add_argument('--pround', type=float, default=0.1)
parser.add_argument('--n_sample', type=int, default=1)
parser.add_argument('--n_point', type=int, default=1024)
args = parser.parse_args()

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
num_iter = int(math.ceil(math.log2(len(param_list))))
per_round_budget = int(math.ceil(len(param_list) / num_iter))

for i in range(1, num_iter + 1):
    split_num = get_split(param_list)
    print('Hyperband step %d, minimum value = %f' % (i, split_num))
    n_hyperband_sample = int(math.ceil(2 * per_round_budget / len(param_list)))
    new_param_list = []
    for param in param_list:
        if param['target'] >= split_num:
            reward = 0
            for _ in range(n_hyperband_sample):
                reward += black_box_class.black_box_reward(**param['params'])
            reward /= n_hyperband_sample
            param['target'] = reward
            new_param_list.append(param)
    param_list = new_param_list

print('Hyperband completed.')
print(param_list[0])
