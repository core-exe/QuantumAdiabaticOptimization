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

b_dict = { 'b_1': 0.1441881510361327,
  'b_2': 0.1468141617959961,
  'b_3': 0.11987021631948974,
  'b_4': -0.10941559013087723,
  'b_5': -0.03496046926672386,
  'b_6': -0.00997377084934975}
b = [x for x in b_dict.keys()]

reward_list = []

for t in np.linspace(0, 62.2, 312):
    target = np.random.randint(0, 2, (6,))
    problem = GroverSearchProblem(6, target)
    hamit_start = QubitOperator('') * 0
    for i in range(6):
        hamit_start += QubitOperator('X{}'.format(i), 1/6)
    hamit = [
        TimeHamitonian(problem.final_hamit, FourierTrojectory((0,1), 62.2, b)),
        TimeHamitonian(hamit_start, FourierTrojectory((1,0), 62.2, b))
    ]
    driver = QADriver(6, hamit, 0.2, t, [0]*6)
    reward = 1 - problem.loss(driver.simulate()[0])
    print(reward)
    reward_list.append(reward)

