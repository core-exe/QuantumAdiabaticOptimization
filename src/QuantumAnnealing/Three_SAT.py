from projectq.ops import QubitOperator
from QACircuitDriver import LinearTrojectory, TimeHamitonian, QADriver
import numpy as np

class ThreeSatProblem:
    def __init__(self, n_qubit, clauses) -> None:
        self.n_qubit = n_qubit
        self.clauses = clauses
        
        self.final_hamit = QubitOperator('') * 0
        for clause in clauses:
            clause_hamit = QubitOperator('')
            for qubit, sign in clause:
                clause_hamit *= 0.5 * (QubitOperator('') - sign * QubitOperator('Z{}'.format(qubit)))
            clause_hamit = QubitOperator('') - clause_hamit
            self.final_hamit += clause_hamit
        self.final_hamit = self.final_hamit / len(clauses)
    
    def loss(self, state_info):
        bitmap, state = state_info
        print(bitmap)
        bitpos_to_qubit = {bitmap[k] : k for k in bitmap.keys()}
        def get_mask(clause):
            mask = np.array([1])
            qubit_to_sign = {c[0]: c[1] for c in clause}
            for i in range(self.n_qubit-1, -1, -1):
                if bitpos_to_qubit[i] not in qubit_to_sign.keys():
                    bit_mask = np.array([1, 1])
                else:
                    sign = qubit_to_sign[bitpos_to_qubit[i]]
                    if sign == 1:
                        bit_mask = np.array([0, 1])
                    else:
                        bit_mask = np.array([1, 0])
                mask = np.tensordot(mask, bit_mask, 0).flatten()
            return 1-mask
        probs = (state * state.conjugate()).real
        clause_expect = 0
        for clause in self.clauses:
            mask = get_mask(clause)
            clause_expect += np.dot(mask, probs)
            print(probs)
            print(mask)
        return 1-clause_expect / len(self.clauses)

def get_default_driver_3sat(problem, time_final, time_step):
    n_qubit = problem.n_qubit
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
    problem = ThreeSatProblem(3, [[(0,1),(1,1),(2,-1)]])
    print(problem.final_hamit)

    driver = get_default_driver_3sat(problem, 32, 1)
    print(problem.loss(driver.simulate()[0]))