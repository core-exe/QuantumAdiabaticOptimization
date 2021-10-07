import gym
import numpy as np
from gym import spaces

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
            s += self.b[m]*np.sin(m*r*np.pi)
        return self.mags[0]*(1-s)+self.mags[1]*s


class GroverSearchEnv(gym.Env):
    def __init__(self, n_qubit, cutoff, time_final, time_step):
        super(GroverSearchEnv, self).__init__()
        self.action_space = spaces.Box(-1, 1, (cutoff,), dtype=np.float32)
        self.observation_space = spaces.MultiBinary(n_qubit)
        self.n_qubit = n_qubit
        self.cutoff = cutoff
        self.time_final = time_final
        self.time_step = time_step
    
    def step(self, action):
        problem = GroverSearchProblem(self.n_qubit, self.target)
        hamit_start = QubitOperator('') * 0
        for i in range(self.n_qubit):
            hamit_start += QubitOperator('X{}'.format(i), 1/self.n_qubit)
        hamit = [
            TimeHamitonian(problem.final_hamit, FourierTrojectory((0,1), self.time_final, action)),
            TimeHamitonian(hamit_start, FourierTrojectory((1,0), self.time_final, action))
        ]
        driver = QADriver(self.n_qubit, hamit, self.time_step, self.time_final, [0]*self.n_qubit)
        reward = 1 - problem.loss(driver.simulate()[0])
        info = {
            'target': self.target,
            'action': action,
            'reward': reward
        }
        return np.zeros(self.n_qubit), reward, True, info

    def reset(self):
        self.target = self.observation_space.sample()
        return np.zeros(self.n_qubit)