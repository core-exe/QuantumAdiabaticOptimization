import numpy as np
from projectq.ops import QubitOperator
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


def get_Fourier_driver_reward(instance, coef, time_final, time_step):
    hamit_start = QubitOperator('') * 0
    for i in range(instance.n_qubit):
        hamit_start += QubitOperator('X{}'.format(i), 1/instance.n_qubit)
    hamit = [
        TimeHamitonian(instance.final_hamit, FourierTrojectory((0,1), time_final, coef)),
        TimeHamitonian(hamit_start, FourierTrojectory((1,0), time_final, coef))
    ]
    driver = QADriver(instance.n_qubit, hamit, time_step, time_final, [0]*instance.n_qubit)
    loss = instance.loss(driver.simulate()[0])
    return 1 - loss


class FourierBlackBox:
    
    def __init__(self, problem, n_qubit, cutoff, time_final, time_step, pround=(-1, 1), num_sample=1):
        self.problem = problem
        self.n_qubit = n_qubit
        self.cutoff = cutoff
        self.time_final = time_final
        self.time_step = time_step
        self.num_sample = num_sample

        name_list = ['b_%d' % i for i in range(1, self.cutoff + 1)]
        value_list = [pround] * self.cutoff
        self.prounds = dict(zip(name_list, value_list))

    def black_box_reward(self, **coef_dict):
        coef = [x for x in coef_dict.values()]
        reward = 0
        for _ in range(self.num_sample):
            instance = self.problem(self.n_qubit)
            reward += get_Fourier_driver_reward(instance, coef, self.time_final, self.time_step)
        return reward / self.num_sample
