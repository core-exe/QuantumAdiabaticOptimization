from projectq import MainEngine
from projectq.ops import QubitOperator, TimeEvolution, H, Rz, Measure, All
from QACircuitBackbone import obtain_example_backbone

def simulate(backbone, debug_mode=False):
    n_qubit = backbone.n_qubit
    eng = MainEngine()
    qubits = eng.allocate_qureg(n_qubit)
    state_init = backbone.state_init
    # state preperation here
    for i in range(n_qubit):
        H | qubits[i]
        Rz(state_init[i]) | qubits[i]
        eng.flush()
        if debug_mode:
            print(eng.backend.cheat())
    while not backbone.is_end():
        #annealing
        hamit, dt = backbone.step()
        hamiltonian = QubitOperator(())
        for pauli, value in hamit:
            hamiltonian += value * QubitOperator(pauli)
        TimeEvolution(time=dt, hamiltonian=hamiltonian) | qubits
        eng.flush()
        if debug_mode:
            print(eng.backend.cheat())
    final_state = eng.backend.cheat()
    All(Measure) | qubits
    eng.flush()
    return final_state

if __name__ == '__main__':
    backbone = obtain_example_backbone()
    simulate(backbone, True)