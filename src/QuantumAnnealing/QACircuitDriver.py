from projectq.ops import QubitOperator, TimeEvolution, H, All, Rz, Measure
from projectq import MainEngine

class Trojectory:
    def __init__(self) -> None:
        pass
    
    def value(self, t):
        return 0

class LinearTrojectory:
    def __init__(self, mags, time_end) -> None:
        self.mags = mags
        self.time_end = time_end
    
    def value(self, t):
        r = t/self.time_end
        return self.mags[0]*(1-r)+self.mags[1]*r

class TimeHamitonian:
    def __init__(self, hamit, troj) -> None:
        self.hamit = hamit
        self.troj = troj
    
    def freeze(self, time):
        return self.hamit, self.troj.value(time)

class QACircuitDriver:
    def __init__(self, n_qubit, hamiltonian, time_step, time_final, state_init) -> None:
        self.eng = MainEngine()
        self.n_qubit = n_qubit
        self.hamiltonian = hamiltonian
        self.t = 0
        self.time_step = time_step
        self.time_final = time_final
        self.state_init = state_init
    
    def prepare(self, qubits):
        All(H) | qubits
        for i in range(self.n_qubit):
            Rz(self.state_init[i]) | qubits[i]
        self.eng.flush()
    
    def step(self, qubits):
        t_end = min(self.t+self.time_step, self.time_final)
        t_mid = 0.5*(self.t+t_end)
        dt = t_end - self.t
        hamit = QubitOperator(())
        for h in self.hamiltonian:
            hamit_static, value = h.freeze(t_mid)
            hamit += hamit_static * value
        self.t = t_end
        TimeEvolution(time = dt, hamiltonian=hamit) | qubits
        self.eng.flush()
    
    def is_end(self):
        return self.t == self.time_final
    
    def get_state(self):
        return self.eng.backend.cheat()
    
    def simulate(self, detail=False):
        state_track = []
        qubits = self.eng.allocate_qureg(self.n_qubit)
        self.prepare(qubits)
        if detail:
            state_track.append((0, self.get_state()))
        while not self.is_end():
            self.step(qubits)
            if detail:
                state_track.append((self.t, self.get_state()))
        final_state = self.get_state()
        All(Measure) | qubits
        self.eng.flush()
        for q in qubits:
            self.eng.deallocate_qubit(q)
        return final_state, state_track

if __name__ == '__main__':
    T_end = 32
    # construct example code:
    # from X0+X1 to (1+Z0)(1+Z1)/4
    hamit_start = (QubitOperator('X0') + QubitOperator('X1'))*0.5
    hamit_final = (QubitOperator('Z0') + QubitOperator(()))*(QubitOperator('Z1') + QubitOperator(()))*0.5**2
    hamit_start_t = TimeHamitonian(hamit_start, LinearTrojectory((1,0), T_end))
    hamit_final_t = TimeHamitonian(hamit_final, LinearTrojectory((0,1), T_end))
    driver = QACircuitDriver(
        n_qubit=2,
        hamiltonian=[hamit_start_t, hamit_final_t],
        time_step=0.1,
        time_final=T_end,
        state_init=[0,0]
    )
    print(driver.simulate(True))