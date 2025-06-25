import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
# from qiskit import IBMQ, transpile
import pandas as pd
import time

def build_schwinger_terms(N: int, m: float, g: float, x: float):
    """
    Return list of (pauli_string, coeff) for N-site Schwinger model 
    in the spin-1/2 quantum link formulation.
    Qubit order: [site_0 ... site_{N-1}, link_0 ... link_{N-2}]
    """
    num_qubits = N + (N - 1)
    terms = []
    # Mass term: m/2 * sum_n (-1)^n Z_site_n
    for n in range(N):
        pauli = ['I'] * num_qubits
        pauli[n] = 'Z'
        coeff = ( (-1)**n ) * m/2
        terms.append(("".join(pauli), coeff))
    # Hopping term: x/2 * sum_n [ X_n X_link X_{n+1} + Y_n X_link Y_{n+1} ]
    for n in range(N - 1):
        # X X X
        pauli_x = ['I'] * num_qubits
        pauli_x[n] = 'X'
        pauli_x[N + n] = 'X'
        pauli_x[n+1] = 'X'
        terms.append(("".join(pauli_x), x/2))
        # Y X Y
        pauli_y = ['I'] * num_qubits
        pauli_y[n] = 'Y'
        pauli_y[N + n] = 'X'
        pauli_y[n+1] = 'Y'
        terms.append(("".join(pauli_y), x/2))
    # Electric term: g^2/2 * sum_links (Z_link/2)^2 
    # For spin-1/2 link, L_z = Z/2 so L_z^2 = I/4, but we include dynamic Z_link term 
    for n in range(N - 1):
        pauli = ['I'] * num_qubits
        pauli[N + n] = 'Z'
        coeff = g**2/4
        terms.append(("".join(pauli), coeff))
    return terms

def apply_pauli_rotation(qc: QuantumCircuit, pauli: str, theta: float):
    """
    Implements exp(-i * theta * P) up to global phase, where P is the Pauli string operator.
    Uses basis rotations and a chain of CNOTs, then RZ.
    """
    num_qubits = len(pauli)
    # Basis change: map X->Z and Y->Z
    for q, p in enumerate(pauli):
        if p == 'X':
            qc.h(q)
        elif p == 'Y':
            qc.sdg(q)
            qc.h(q)
    # Identify qubits where p != 'I'
    non_id = [i for i, p in enumerate(pauli) if p != 'I']
    if non_id:
        target = non_id[-1]
        # Entangle others into target
        for ctrl in non_id[:-1]:
            qc.cx(ctrl, target)
        # Z-rotation
        qc.rz(2 * theta, target)
        # Undo entanglement
        for ctrl in reversed(non_id[:-1]):
            qc.cx(ctrl, target)
    # Undo basis change
    for q, p in enumerate(pauli):
        if p == 'X':
            qc.h(q)
        elif p == 'Y':
            qc.h(q)
            qc.s(q)

def trotter_circuit(N: int, m: float, g: float, x: float, time: float, steps: int):
    """
    Build first-order Trotterized evolution circuit e^{-i H t} for the Schwinger Hamiltonian.
    """
    num_qubits = N + (N - 1)
    qc = QuantumCircuit(num_qubits)
    dt = time / steps
    terms = build_schwinger_terms(N, m, g, x)
    for _ in range(steps):
        for pauli, coeff in terms:
            apply_pauli_rotation(qc, pauli, coeff * dt)
    return qc

def prepare_initial_state(qc: QuantumCircuit, N: int, kind: str = 'vacuum', pair_sites: tuple = (0,1)):
    """
    Prepare initial fermionic state on the site qubits:
      'vacuum': staggered vacuum (even sites occupied)
      'pair': particle-antiparticle at pair_sites (relative to vacuum)
    """
    # Vacuum: even sites |1>
    if kind == 'vacuum':
        for i in range(N):
            if i % 2 == 0:
                qc.x(i)
    elif kind == 'pair':
        prepare_initial_state(qc, N, 'vacuum')
        for i in pair_sites:
            qc.x(i)
    else:
        raise ValueError(f"Unknown initial state: {kind}")

def simulate(circ: QuantumCircuit, shots: int = 1024):
    """
    Simulate circuit: if measurements present, return counts; 
    otherwise return Statevector.
    """
    if circ.count_ops().get('measure', 0) > 0:
        sim = AerSimulator()
        job = sim.run(circ, shots=shots)
        return job.result().get_counts()
    else:
        return Statevector.from_instruction(circ)

def measure_z_expectation(circ: QuantumCircuit, qubits: list, shots: int = 1024):
    """
    Measure Z expectation <Z> for each qubit index in qubits.
    Returns dict mapping qubit->expectation.
    """
    meas = circ.copy()
    meas.measure(range(len(meas.qubits)), range(len(meas.qubits)))
    counts = simulate(meas, shots)
    exps = {}
    total = sum(counts.values())
    for q in qubits:
        exp = 0
        for bitstr, cnt in counts.items():
            # Qiskit bitstring has qubit0 as least significant bit
            bit = int(bitstr[::-1][q])
            exp += (1 - 2*bit) * cnt
        exps[q] = exp / total
    return exps

def chiral_condensate(expectations: dict, N: int):
    """
    Compute chiral condensate <psi_bar psi> = (1/N) sum_n (-1)^n <Z_n>
    """
    return sum(((-1)**n) * expectations[n] for n in range(N)) / N

# --- Calibration and benchmarking utilities ---
class ReadoutMitigator:
    def __init__(self):
        self.cal_matrix = None
        self.states = None

    def calibrate(self, backend, qubit_count, shots=1024):
        from qiskit import assemble
        self.states = [format(i, f'0{qubit_count}b') for i in range(2**qubit_count)]
        cal_circuits = []
        for state in self.states:
            qc_cal = QuantumCircuit(qubit_count, qubit_count)
            for q, bit in enumerate(state):
                if bit == '1':
                    qc_cal.x(q)
            qc_cal.measure(range(qubit_count), range(qubit_count))
            qct = transpile(qc_cal, backend)
            cal_circuits.append(qct)
        job = backend.run(assemble(cal_circuits, backend=backend, shots=shots))
        results = job.result()
        import numpy as np
        mat = np.zeros((2**qubit_count, 2**qubit_count))
        for i, state in enumerate(self.states):
            counts = results.get_counts(i)
            for bitstr, cnt in counts.items():
                j = self.states.index(bitstr)
                mat[j, i] = cnt/shots
        self.cal_matrix = mat

    def apply(self, counts):
        import numpy as np
        qubit_count = len(next(iter(counts)))
        probs = np.zeros(2**qubit_count)
        for bitstr, cnt in counts.items():
            idx = self.states.index(bitstr)
            probs[idx] = cnt
        probs /= probs.sum()
        mitigated = np.linalg.inv(self.cal_matrix).dot(probs)
        mitigated = np.clip(mitigated, 0, None)
        mitigated /= mitigated.sum()
        return {self.states[i]: mitigated[i] for i in range(len(self.states))}

def hellinger_dist(p, q):
    import numpy as np
    keys = set(p.keys()).union(q.keys())
    return np.sqrt(1 - sum(np.sqrt(p.get(k,0)*q.get(k,0)) for k in keys))

def compute_z_errors(ideal_exps, hardware_exps):
    return {n: hardware_exps.get(n,0) - ideal_exps.get(n,0) for n in ideal_exps}

class BenchmarkRunner:
    def __init__(self, backend, mitigator=None, shots=1024):
        self.backend = backend
        self.mitigator = mitigator
        self.shots = shots

    def run(self, qc):
        qct = transpile(qc, self.backend, optimization_level=1)
        job = self.backend.run(qct, shots=self.shots)
        raw_counts = job.result().get_counts()
        return self.mitigator.apply(raw_counts) if self.mitigator else raw_counts

    def compare_with_sim(self, qc):
        # ideal probabilities from statevector
        sv = simulate(qc)
        import numpy as np
        probs = {format(i, f'0{qc.num_qubits}b'): abs(amp)**2 for i, amp in enumerate(sv.data)}
        hard_counts = self.run(qc)
        total = sum(hard_counts.values())
        hard_probs = {k: v/total for k,v in hard_counts.items()}
        # Observable errors
        ideal_z = measure_z_expectation(qc, list(range(qc.num_qubits)), shots=self.shots)
        hw_z = {}
        for n in range(qc.num_qubits):
            exp = 0
            for bitstr, prob in hard_probs.items():
                bit = int(bitstr[::-1][n])
                exp += (1 - 2*bit) * prob
            hw_z[n] = exp
        return {
            'hellinger': hellinger_dist(probs, hard_probs),
            'z_errors': compute_z_errors(ideal_z, hw_z),
        }

def run_parameter_sweep(param_list, runner):
    results = []
    for params in param_list:
        N = params['N']; m=params['m']; g=params['g']; x=params['x']
        t=params['t']; steps=params['steps']; shots=params.get('shots',1024)
        num_q = N + (N-1)
        qc = QuantumCircuit(num_q, num_q)
        prepare_initial_state(qc, N, kind=params.get('kind','vacuum'),
                              pair_sites=params.get('pair_sites',(0,1)))
        qc.compose(trotter_circuit(N,m,g,x,t,steps), inplace=True)
        metrics = runner.compare_with_sim(qc)
        results.append({**params, **metrics})
    df = pd.DataFrame(results)
    df.to_csv('benchmark_results.csv', index=False)
    return df

if __name__ == '__main__':
    # Example usage
    N = 4
    m, g, x = 1.0, 1.0, 1.0
    t, steps = 10.0, 5
    shots = 1024

    # Full circuit: initial state + evolution + measurement
    num_qubits = N + (N - 1)
    qc_full = QuantumCircuit(num_qubits, num_qubits)
    
    # Prepare initial state and inspect
    prepare_initial_state(qc_full, N, kind='pair', pair_sites=(1,2))
    psi0 = Statevector.from_instruction(qc_full)
    print("Initial state:", psi0)
    
    # Trotter evolution appended onto the same circuit
    evo_qc = trotter_circuit(N, m, g, x, t, steps)
    qc_full.compose(evo_qc, inplace=True)
    
    # Simulate full circuit
    final = simulate(qc_full)
    print("Final state:", final)
    
    # Measure observables on the full circuit
    exps = measure_z_expectation(qc_full, list(range(N)))
    print("Site Z expectations:", exps)
    print("Chiral condensate:", chiral_condensate(exps, N))

    # === Benchmark on QPU ===
    # IBMQ.load_account()
    # provider = IBMQ.get_provider(hub='your_hub', group='your_group', project='your_project')
    # backend = provider.get_backend('your_qpu_name')
    # mitigator = ReadoutMitigator()
    # mitigator.calibrate(backend, num_qubits, shots=shots)
    # runner = BenchmarkRunner(backend, mitigator, shots=shots)
    # metrics = runner.compare_with_sim(qc_full)
    # print("Benchmark metrics:", metrics)
