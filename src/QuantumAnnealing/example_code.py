from projectq import MainEngine  # import the main compiler engine
from projectq.ops import H, Measure  # import the operations we want to perform (Hadamard and measurement)

eng = MainEngine()  # create a default compiler (the back-end is a simulator)
qubit = eng.allocate_qubit()  # allocate 1 qubit

H | qubit  # apply a Hadamard gate

eng.flush()  # flush all gates (and execute measurements)
print(eng.backend.cheat())

Measure | qubit  # measure the qubit
print(eng.backend.cheat())
