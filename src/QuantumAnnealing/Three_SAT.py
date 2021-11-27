from projectq.ops import QubitOperator
from .QACircuitDriver import LinearTrojectory, TimeHamitonian, QADriver
import numpy as np

def get_3sat_clauses(n_vars=4, n_clauses=17):
    rand_var = np.random.randint(low=0, high=n_vars, size=(n_clauses, 3))
    rand_positive = np.random.randint(low=0, high=2, size=(n_clauses, 3))*2-1
    clauses = []
    for i in range(n_clauses):
        clauses.append([(int(rand_var[i][j]), int(rand_positive[i][j])) for j in range(3)])
    return clauses

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
        state = np.array(state)
        # print(bitmap)
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
        mask_total = np.zeros((2**self.n_qubit, ))
        probs = (state * state.conjugate()).real
        for clause in self.clauses:
            mask_total += get_mask(clause)
        clause_expect = np.dot(mask_total, probs)
            # print(probs)
            # print(mask)
        return 1-clause_expect / np.max(mask_total)

def get_3sat_problem(n_qubit):
    n_clauses = int(4.25 * n_qubit)
    clauses = get_3sat_clauses(n_qubit, n_clauses)
    return ThreeSatProblem(n_qubit, clauses)

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