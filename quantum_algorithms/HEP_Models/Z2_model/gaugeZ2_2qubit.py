from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.quantum_info import Statevector, Pauli
import numpy as np
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

# Parameters
J = 1.0
lam = 1.0
t_max = 10
n_steps = 100
times = np.linspace(0, t_max, n_steps)

# Initialize service (uncomment and use your token if needed)
# QiskitRuntimeService.save_account(
#     channel="ibm_quantum", 
#     token="YOUR_TOKEN_HERE",
#     overwrite=True
# )

service = QiskitRuntimeService()
backend = service.least_busy(simulator=False, operational=True)
print(f"Using backend: {backend.name}")

# Qubit roles:
# q0 = ψ₀ (matter at site 0)
# q1 = U₀ (link between site 0–1)
# q2 = ψ₁ (virtual, inferred from Gauss law, not encoded directly)
# q3 = U₁ (link between site 1–2)
# q4 = ψ₂ (matter at site 2)

def build_circuit(t, J, lam, measure=True):
    """Build the quantum circuit for Z₂ lattice gauge theory evolution."""
    # Create circuit - simpler approach
    if measure:
        qc = QuantumCircuit(5, 5)
    else:
        qc = QuantumCircuit(5)

    # Initial state: All in |1⟩
    qc.x(0)  # ψ₀
    qc.x(1)  # U₀
    qc.x(3)  # U₁
    qc.x(4)  # ψ₂

    # Kinetic energy: flip all dynamical fields
    for q in [0, 1, 3, 4]:
        qc.rx(2 * J * t, q)

    # Interaction terms:
    qc.rzz(2 * lam * t, 0, 1)  # Zψ₀ ZU₀
    qc.rzz(2 * lam * t, 1, 3)  # ZU₀ ZU₁ — analog of Zψ₁ included via constraint
    qc.rzz(2 * lam * t, 3, 4)  # ZU₁ Zψ₂

    if measure:
        qc.measure_all()
    
    return qc

def expval_z(counts, qubit, shots):
    """Calculate expectation value of Z operator for a given qubit."""
    exp = 0
    for bitstr, cnt in counts.items():
        bit = int(bitstr[::-1][qubit])
        exp += (1 - 2 * bit) * cnt
    return exp / shots

def expval_zz(counts, q1, q2, shots):
    """Calculate expectation value of ZZ operator for two qubits."""
    exp = 0
    for bitstr, cnt in counts.items():
        b1 = int(bitstr[::-1][q1])
        b2 = int(bitstr[::-1][q2])
        exp += (1 - 2 * b1) * (1 - 2 * b2) * cnt
    return exp / shots

# Initialize storage for results
z_expect = [[] for _ in range(5)]
zz_01, zz_13, zz_34 = [], [], []
g0_expect, g1_expect, g2_expect = [], [], []

shots = 1024

# Create the sampler with the backend
sampler = Sampler(backend)

# First, let's do a single test run to verify everything works
print("\nPerforming test run to verify setup...")
test_qc = build_circuit(times[0], J, lam, measure=True)
test_transpiled = transpile(test_qc, backend)
print(f"Test circuit depth: {test_transpiled.depth()}")
print(f"Test circuit gates: {test_transpiled.count_ops()}")

# Run test
test_job = sampler.run([test_transpiled], shots=10)
test_result = test_job.result()

# Check the result structure
pub_result = test_result[0]
print(f"Classical register names in data: {[attr for attr in dir(pub_result.data) if not attr.startswith('_')]}")

# Now run the actual simulation
print(f"\nRunning {n_steps} time steps with {shots} shots each...")
print("This will use approximately", n_steps * shots / 1000, "kilo-shots of your monthly credits.")

for i, t in enumerate(times):
    if i % 10 == 0:
        print(f"Progress: {i}/{n_steps} time steps completed")
    
    # Build circuit
    qc = build_circuit(t, J, lam, measure=True)
    
    # Transpile the circuit for the backend
    transpiled_qc = transpile(qc, backend)
    
    # Run the circuit
    job = sampler.run([transpiled_qc], shots=shots)
    result = job.result()
    
    # Extract the counts from the result
    pub_result = result[0]
    
    # Access the measurement data - it might be in 'c' or 'meas'
    if hasattr(pub_result.data, 'c'):
        counts = pub_result.data.c.get_counts()
    elif hasattr(pub_result.data, 'meas'):
        counts = pub_result.data.meas.get_counts()
    else:
        # If neither works, print what's available and try the first one
        attrs = [attr for attr in dir(pub_result.data) if not attr.startswith('_')]
        print(f"Available attributes: {attrs}")
        counts = getattr(pub_result.data, attrs[0]).get_counts()
    
    # Calculate expectation values
    for j in range(5):
        z_expect[j].append(expval_z(counts, j, shots))
    
    zz_01.append(expval_zz(counts, 0, 1, shots))
    zz_13.append(expval_zz(counts, 1, 3, shots))
    zz_34.append(expval_zz(counts, 3, 4, shots))
    
    # Gauss law operators (same as ZZ operators in this encoding)
    g0_expect.append(expval_zz(counts, 0, 1, shots))
    g1_expect.append(expval_zz(counts, 1, 3, shots))
    g2_expect.append(expval_zz(counts, 3, 4, shots))

print("Simulation completed!")

# --------------------------
# Plotting
# --------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Individual Z expectations
labels = ['ψ₀', 'U₀', 'ψ₁ (implicit)', 'U₁', 'ψ₂']
colors = ['tab:blue', 'tab:orange', 'gray', 'tab:green', 'tab:red']

for i in range(5):
    if i != 2:  # Skip implicit qubit
        ax1.plot(times, z_expect[i], label=f"⟨Z_{labels[i]}⟩", color=colors[i], linewidth=2)

ax1.set_xlabel("Time")
ax1.set_ylabel("⟨Z⟩ Expectation value")
ax1.set_title("Z₂ Lattice Gauge Theory: Single Qubit Expectations")
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best')

# Plot 2: ZZ correlations and Gauss laws
ax2.plot(times, zz_01, label="⟨Zψ₀ ZU₀⟩", linestyle='-', linewidth=2, color='purple')
ax2.plot(times, zz_13, label="⟨ZU₀ ZU₁⟩", linestyle='-', linewidth=2, color='brown')
ax2.plot(times, zz_34, label="⟨ZU₁ Zψ₂⟩", linestyle='-', linewidth=2, color='pink')

# Gauss laws (should be conserved)
ax2.plot(times, g0_expect, label="Gauss G₀", linestyle='--', linewidth=2, color='black', alpha=0.7)
ax2.plot(times, g1_expect, label="Gauss G₁", linestyle='--', linewidth=2, color='darkgray', alpha=0.7)
ax2.plot(times, g2_expect, label="Gauss G₂", linestyle='--', linewidth=2, color='lightgray', alpha=0.7)

ax2.set_xlabel("Time")
ax2.set_ylabel("⟨ZZ⟩ Expectation value")
ax2.set_title("Z₂ Lattice Gauge Theory: Correlations and Gauss Law Conservation")
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best')

plt.tight_layout()
plt.savefig('z2_gauge_simulation.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate and print Gauss law violations
print("\nGauss Law Conservation Check:")
g0_violation = np.std(g0_expect)
g1_violation = np.std(g1_expect)
g2_violation = np.std(g2_expect)
print(f"G₀ standard deviation: {g0_violation:.4f}")
print(f"G₁ standard deviation: {g1_violation:.4f}")
print(f"G₂ standard deviation: {g2_violation:.4f}")

# Additional analysis
print("\nTime-averaged expectation values:")
for i, label in enumerate(labels):
    if i != 2:
        avg = np.mean(z_expect[i])
        std = np.std(z_expect[i])
        print(f"⟨Z_{label}⟩ = {avg:.3f} ± {std:.3f}")

# Calculate total shots used
total_shots = n_steps * shots
print(f"\nTotal shots used: {total_shots:,}")
print(f"Approximate cost: {total_shots / 1_000_000:.2f} mega-shots")

# Save results for future analysis
results = {
    'times': times,
    'z_expect': z_expect,
    'zz_correlations': {
        'zz_01': zz_01,
        'zz_13': zz_13,
        'zz_34': zz_34
    },
    'gauss_laws': {
        'g0': g0_expect,
        'g1': g1_expect,
        'g2': g2_expect
    },
    'parameters': {
        'J': J,
        'lambda': lam,
        't_max': t_max,
        'n_steps': n_steps,
        'shots': shots,
        'backend': backend.name
    }
}

# Optional: Save to file
import json
with open('z2_gauge_results.json', 'w') as f:
    # Convert numpy arrays to lists for JSON serialization
    json_results = {
        'times': times.tolist(),
        'z_expect': [arr for arr in z_expect],
        'zz_correlations': {k: v for k, v in results['zz_correlations'].items()},
        'gauss_laws': {k: v for k, v in results['gauss_laws'].items()},
        'parameters': results['parameters']
    }
    json.dump(json_results, f, indent=2)

print("\nResults saved to z2_gauge_results.json")