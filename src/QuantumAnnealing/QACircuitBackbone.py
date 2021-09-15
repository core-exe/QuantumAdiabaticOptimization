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

class Hamiltonian:
    def __init__(self, components) -> None:
        # components in form of [("X0", troj), ...]
        # usually use 2-local
        self.components = components
    
    def freeze(self, time):
        hamit = []
        for pauli, troj in self.components:
            hamit.append((pauli, troj.value(time)))
        return hamit

class CircuitBackbone:
    def __init__(self, n_qubit, hamiltonian, time_step, time_final, state_init) -> None:
        self.n_qubit = n_qubit
        self.hamiltonian = hamiltonian
        self.t = 0
        self.time_step = time_step
        self.time_final = time_final
        self.state_init = state_init
    
    def step(self):
        t_end = min(self.t+self.time_step, self.time_final)
        t_mid = 0.5*(self.t+t_end)
        dt = t_end - self.t
        self.t = t_end
        return self.hamiltonian.freeze(t_mid), dt
    
    def is_end(self):
        return self.t == self.time_final

def obtain_example_backbone(T_step=0.1, T_end=32):
    hamit = Hamiltonian([("X0", LinearTrojectory((1,0), T_end)), ("Z0", LinearTrojectory((0, 1), T_end))])
    backbone = CircuitBackbone(1, hamit, T_step, T_end, [0])
    return backbone

if __name__ == '__main__':
    T_end = 32
    hamit = Hamiltonian([("X0", LinearTrojectory((1,0), T_end)), ("Z0", LinearTrojectory((0, 1), T_end))])
    backbone = CircuitBackbone(1, hamit, 1, T_end, [0])
    while not backbone.is_end():
        print(backbone.step())