from projectq.ops import QubitOperator
from .QACircuitDriver import LinearTrojectory, TimeHamitonian, QADriver

class GroverSearchProblem:
    def __init__(self, n_qubit, target=None) -> None:
        self.n_qubit = n_qubit
        if target is None:
            target = [0]*self.n_qubit
        self.target = target
        self.final_hamit = QubitOperator('') * 1
        for i in range(n_qubit):
            if self.target[i] == 0:
                self.final_hamit *= 0.5 * (QubitOperator('') + QubitOperator('Z{}'.format(i)))
            elif self.target[i] == 1:
                self.final_hamit *= 0.5 * (QubitOperator('') - QubitOperator('Z{}'.format(i)))
    
    def loss(self, state_info):
        bitmap, state = state_info
        entry = sum([2**bitmap[i]*self.target[i] for i in range(self.n_qubit)])
        amp = state[entry]
        prob = (amp*amp.conjugate()).real
        return 1-prob

def get_default_driver_gs(problem, n_qubit, time_final, time_step, target=None):
    hamit_start = QubitOperator('') * 0
    for i in range(n_qubit):
        hamit_start += QubitOperator('X{}'.format(i), 1/n_qubit)
    hamit = [
        TimeHamitonian(problem.final_hamit, LinearTrojectory((0,1), time_final)),
        TimeHamitonian(hamit_start, LinearTrojectory((1,0), time_final))
    ]
    driver = QADriver(n_qubit, hamit, time_step, time_final, [0]*n_qubit)
    return driver

if __name__ == '__main__':
    problem = GroverSearchProblem(2, [0,0])
    print(problem.final_hamit)

    driver = get_default_driver_gs(problem, 2, 32, 0.2)
    print(problem.loss(driver.simulate()[0]))
