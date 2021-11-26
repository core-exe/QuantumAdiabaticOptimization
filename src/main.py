from black_box import FourierBlackBox
from bayes_opt import BayesianOptimization
from QuantumAnnealing.Three_SAT import get_3sat_problem
from QuantumAnnealing.GroverSearch import get_gs_problem

black_box_class = FourierBlackBox(get_3sat_problem, 
                                  n_qubit=4, 
                                  cutoff=6, 
                                  time_final=62.2, 
                                  time_step=0.2, 
                                  pround=(-0.1, 0.1), 
                                  num_sample=10)
optimizer = BayesianOptimization(black_box_class.black_box_reward, 
                                 black_box_class.prounds,
                                 verbose=2)

print('Optimization started.')

optimizer.probe([0] * black_box_class.cutoff)
optimizer.maximize(init_points=0, n_iter=1000)

print(optimizer.max)