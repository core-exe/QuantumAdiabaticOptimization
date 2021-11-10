import numpy as np
from projectq.ops import QubitOperator
from QuantumAnnealing.GroverSearch import GroverSearchProblem
from QuantumAnnealing.QACircuitDriver import TimeHamitonian, QADriver


class FourierTrojectory:

    def __init__(self, mags, time_end, b):
        self.mags = mags
        self.time_end = time_end
        self.b = b
    
    def value(self, t):
        r = t/self.time_end
        s = r
        for m in range(len(self.b)):
            s += self.b[m]*np.sin((m+1)*r*np.pi)
        return self.mags[0]*(1-s)+self.mags[1]*s


class BlackBoxClass:
    
    def __init__(self, n_qubit, cutoff, time_final, time_step, pround=(-1, 1), num_sample=10):
        self.n_qubit = n_qubit
        self.cutoff = cutoff
        self.time_final = time_final
        self.time_step = time_step
        self.num_sample = num_sample

        name_list = ['b_%d' % i for i in range(1, self.cutoff + 1)]
        value_list = [pround] * self.cutoff
        self.prounds = dict(zip(name_list, value_list))

    def black_box(self, **b_dict):
        b = [x for x in b_dict.values()]
        reward = 0
        for _ in range(self.num_sample):
            target = np.random.randint(0, 2, (self.n_qubit,))
            problem = GroverSearchProblem(self.n_qubit, target)
            hamit_start = QubitOperator('') * 0
            for i in range(self.n_qubit):
                hamit_start += QubitOperator('X{}'.format(i), 1/self.n_qubit)
            hamit = [
                TimeHamitonian(problem.final_hamit, FourierTrojectory((0,1), self.time_final, b)),
                TimeHamitonian(hamit_start, FourierTrojectory((1,0), self.time_final, b))
            ]
            driver = QADriver(self.n_qubit, hamit, self.time_step, self.time_final, [0]*self.n_qubit)
            reward += 1 - problem.loss(driver.simulate()[0])
        return reward / self.num_sample
