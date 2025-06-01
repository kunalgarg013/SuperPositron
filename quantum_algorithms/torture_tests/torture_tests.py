from qiskit import QuantumCircuit
from numpy import pi
import json

# Dictionary to hold all circuits and their metadata
circuit_tests = {}

# Test 1: Overrotation Amplifier (RX)
qc1 = QuantumCircuit(1, 1)
for _ in range(16):
    qc1.rx(pi/4, 0)
qc1.measure(0, 0)
circuit_tests["Overrotation_Amplifier"] = qc1.qasm()

# Test 2: T and T† cancellation stress
qc2 = QuantumCircuit(1, 1)
qc2.h(0)
for _ in range(50):
    qc2.t(0)
    qc2.tdg(0)
qc2.h(0)
qc2.measure(0, 0)
circuit_tests["T_Tdag_Cancellation"] = qc2.qasm()

# Test 3: Commutator Loop
qc3 = QuantumCircuit(1, 1)
for _ in range(10):
    qc3.rx(pi/2, 0)
    qc3.rz(pi/2, 0)
    qc3.rx(-pi/2, 0)
    qc3.rz(-pi/2, 0)
qc3.measure(0, 0)
circuit_tests["Commutator_Loop"] = qc3.qasm()

# Test 4: Delay-Based Drift Test
qc4 = QuantumCircuit(1, 1)
qc4.h(0)
qc4.delay(100, 0, 'ns')  # Adjust if delay units differ on your backend
qc4.h(0)
qc4.measure(0, 0)
circuit_tests["Delay_Drift_Test"] = qc4.qasm()

# Test 5: Crosstalk Pinger
qc5 = QuantumCircuit(2, 2)
qc5.x(0)
qc5.h(1)
qc5.x(0)
qc5.h(1)
qc5.measure([0, 1], [0, 1])
circuit_tests["Crosstalk_Pinger"] = qc5.qasm()

# Test 6: Basis Mismatch (RX)
qc6 = QuantumCircuit(1, 1)
qc6.rx(pi/2, 0)
qc6.measure(0, 0)
circuit_tests["Basis_Mismatch_RX"] = qc6.qasm()

# Output all QASM circuits to JSON
with open("qc_torture_tests.json", "w") as f:
    json.dump(circuit_tests, f, indent=2)

print("QASM circuits saved to qc_torture_tests.json")
