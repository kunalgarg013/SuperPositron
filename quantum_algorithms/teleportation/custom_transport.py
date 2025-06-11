from math import sqrt, pi
import cmath
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, state_fidelity, partial_trace

def custom_initializer(qc: QuantumCircuit):
    """
    Prepare a custom state |ψ⟩ = sqrt(0.3)|0⟩ + sqrt(0.7)e^{iπ/3}|1⟩ on qubit 0.
    """
    amp_0 = sqrt(0.3)
    amp_1 = sqrt(0.7) * cmath.exp(1j * pi / 3)
    qc.initialize([amp_0, amp_1], 0)

def create_teleportation_circuit(init_fn=None, measure_basis='z', measure_final=True) -> QuantumCircuit:
    """
    Creates a teleportation circuit with optional custom initialization and measurement basis.
    
    Args:
        init_fn: Function to initialize qubit 0 in a desired state
        measure_basis: 'x', 'y', or 'z' basis for final qubit measurement
        measure_final: Whether to include measurement of final qubit
        
    Returns:
        QuantumCircuit
    """
    qr = QuantumRegister(3, 'q')  # q0: state, q1: ent_A, q2: ent_B
    cr = ClassicalRegister(3, 'c')  # c[0,1]: Bell result, c[2]: final measurement
    qc = QuantumCircuit(qr, cr)

    # Custom state prep
    if init_fn:
        init_fn(qc)
    else:
        qc.h(0)
        qc.t(0)

    # Bell pair preparation between q1 and q2
    qc.h(1)
    qc.cx(1, 2)

    # Bell measurement on q0 and q1
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.measure(1, 1)

    # Conditional correction on q2
    with qc.if_test((cr[1], 1)):
        qc.x(2)
    with qc.if_test((cr[0], 1)):
        qc.z(2)

    # Optional final measurement
    if measure_final:
        if measure_basis == 'y':
            qc.sdg(2)
            qc.h(2)
        elif measure_basis == 'x':
            qc.h(2)
        qc.measure(2, 2)

    return qc

def run_teleportation_and_fidelity(qc: QuantumCircuit) -> float:
    """
    Runs the circuit on AerSimulator with statevector and computes teleportation fidelity.
    
    Returns:
        Fidelity between expected and teleported state
    """
    sim = AerSimulator(method='statevector')
    qc.save_statevector()
    transpiled = transpile(qc, backend=sim)
    result = sim.run(transpiled).result()
    statevec = Statevector(result.get_statevector())

    # Compare teleported state on qubit 2 with expected
    ideal = Statevector([sqrt(0.3), sqrt(0.7) * cmath.exp(1j * pi / 3)])
    teleported = partial_trace(statevec, [0, 1])
    fidelity = state_fidelity(ideal, teleported)
    return fidelity

if __name__ == "__main__":
    qc = create_teleportation_circuit(
        init_fn=custom_initializer,
        measure_basis='y',
        measure_final=False
    )
    fidelity = run_teleportation_and_fidelity(qc)
    print("Teleportation fidelity (custom state):", fidelity)
