# Full teleportation module with optional basis measurement and fidelity estimation

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import (
    Statevector, 
    state_fidelity, 
    partial_trace
)

def create_teleportation_circuit(measure_basis: str = 'z', measure_final: bool = True) -> QuantumCircuit:
    """
    Creates a quantum teleportation circuit with optional basis measurement.
    
    Args:
        measure_basis: Measurement basis ('x', 'y', or 'z')
        measure_final: Whether to include final measurement
        
    Returns:
        QuantumCircuit: Configured teleportation circuit
    """
    qr = QuantumRegister(3, 'q')  # q0: psi, q1: ent_A, q2: ent_B
    cr = ClassicalRegister(3, 'c')  # c0, c1: Bell measurements; c2: final qubit
    qc = QuantumCircuit(qr, cr)

    # Prepare arbitrary state |ψ⟩ on q0
    qc.h(0)
    qc.t(0)

    # Create Bell pair between q1 and q2
    qc.h(1)
    qc.cx(1, 2)

    # Bell measurement on q0 and q1
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.measure(1, 1)

    # Conditional corrections on q2 using the new syntax
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

def run_teleportation_circuit(qc, backend_name="aer_simulator", shots=1024, return_statevector=False):
    try:
        if backend_name == "aer_simulator":
            backend = AerSimulator(
                method='statevector' if return_statevector else 'automatic'
            )
        else:
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService()
            backend = service.backend(backend_name)
    except ImportError as e:
        raise ImportError(
            "Missing dependencies. Please run:\n"
            "pip install qiskit qiskit-aer qiskit-ibm-runtime"
        ) from e

    qc_compiled = transpile(qc, backend, optimization_level=1)
    
    if return_statevector and backend_name == "aer_simulator":
        qc_compiled.save_statevector()
    
    try:
        job = backend.run(qc_compiled, shots=shots)
        result = job.result()
        counts = result.get_counts()

        if return_statevector and backend_name == "aer_simulator":
            statevector = result.get_statevector()
            final_sv = Statevector(statevector)
            return counts, final_sv
        return counts
    except Exception as e:
        raise RuntimeError(f"Circuit execution failed: {str(e)}") from e

def calculate_teleportation_fidelity(final_sv):
    # Expected state: T.H|0> = |+i⟩ = (|0⟩ + i|1⟩)/√2
    ideal_circuit = QuantumCircuit(1)
    ideal_circuit.h(0)
    ideal_circuit.t(0)
    ideal_sv = Statevector.from_instruction(ideal_circuit)

    # Extract state of q2 (index 2) using partial_trace
    # Keep qubit 2, trace out qubits 0 and 1
    teleported_sv = partial_trace(final_sv, [0, 1])
    fidelity = state_fidelity(ideal_sv, teleported_sv)
    return fidelity


# qc = create_teleportation_circuit(measure_final=False)  # skip final measurement
# counts, final_sv = run_teleportation_circuit(qc, return_statevector=True)

qc = create_teleportation_circuit(measure_basis='y', measure_final=True)
counts = run_teleportation_circuit(qc, return_statevector=False)
print("Counts:  ", counts)

# fidelity = calculate_teleportation_fidelity(final_sv)

# print("Teleportation fidelity:", fidelity)
