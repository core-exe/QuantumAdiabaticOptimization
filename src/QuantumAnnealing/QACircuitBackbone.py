class Trojectory:
    def __init__(self) -> None:
        pass
    
    def value(self, time):
        return 0

class LinearTrojectory:
    def __init__(self, mags, time_end) -> None:
        self.mags = mags
        self.time_end = time_end
    
    def value(self, t):
        r = t/self.time_end
        return self.mags[0]*(1-r)+self.mags[1]*r

class Pauli:
    def __init__(self, type, target, trojectory) -> None:
        self.type = type
        assert type in ['x', 'y', 'z'] + ['xx', 'yy', 'zz', 'xy', 'yz', 'xz']
        self.target = target
        self.trojectory = trojectory
    
    def freeze(self, time):
        return {'type': self.type, 'target': self.target, 'mag': self.trojectory.value(time)}

class Hamiltonian:
    # only 2-local
    def __init__(self, n_qubits, pauli):
        self.n_qubits = n_qubits
        self.pauli = pauli

    def freeze(self, time):
        pauli_list = []
        for p in self.pauli:
            pauli_list.append(p.freeze(time))
        return pauli_list

class QACircuitBackbone:
    def __init__(self, n_qubits, hamit, time_end):
        self.n_qubits = n_qubits
        self.hamit = hamit
        self.time_end = time_end
    
    def compile(self):
        t = 0
        gates = []
        for t in range(self.time_end):
            gates.append(self.hamit.freeze(t+0.5))
        return gates

if __name__ == '__main__':
    sx = Pauli('x', 0, LinearTrojectory((0.05, 0), 128))
    sz = Pauli('z', 0, LinearTrojectory((0, 0.05), 128))
    paulis = [sx, sz]
    hamit = Hamiltonian(1, paulis)
    circuit_backbone = QACircuitBackbone(1, hamit, 128)
    print(circuit_backbone.compile())