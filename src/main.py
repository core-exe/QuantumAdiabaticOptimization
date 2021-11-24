from black_box import BlackBoxClass
from bayes_opt import BayesianOptimization


black_box_class = BlackBoxClass(n_qubit=5, cutoff=6, time_final=32, time_step=0.2, pround=(-0.5, 0.5), num_sample=1)
optimizer = BayesianOptimization(black_box_class.black_box, black_box_class.prounds)

optimizer.maximize(init_points=10, n_iter=100)
