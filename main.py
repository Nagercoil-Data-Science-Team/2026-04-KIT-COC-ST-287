import matplotlib
matplotlib.use('Agg')  

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import warnings
warnings.filterwarnings("ignore")

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import StatevectorEstimator

from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA

# ==============================
# Output directory for all plots
# ==============================
OUTPUT_DIR = "vqe_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def savefig(filename, fig=None):
    """Save figure to OUTPUT_DIR and close it."""
    path = os.path.join(OUTPUT_DIR, filename)
    if fig is not None:
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close('all')
    print(f"  [SAVED] {path}")

# ==============================
# Plot Settings
# ==============================
plt.rcParams['font.family'] = 'DejaVu Serif'   # Fallback safe font
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'

# ==============================
# STEP 1: H2 Hamiltonian (2 qubit)
# ==============================
hamiltonian_2q = SparsePauliOp.from_list([
    ("ZZ", -1.05),
    ("IX", 0.39),
    ("XI", 0.39),
    ("II", -0.01)
])

print("\n===== H2 Hamiltonian =====")
print(hamiltonian_2q)

# ==============================
# STEP 2: Ansatz (2 qubit)
# ==============================
ansatz_2q = TwoLocal(
    num_qubits=2,
    rotation_blocks='ry',
    entanglement_blocks='cx',
    reps=3
)

print("\n===== Ansatz Circuit =====")
print(ansatz_2q)

fig = ansatz_2q.draw(output='mpl', scale=1.5, fold=20)
fig.set_size_inches(14, 8)
fig.suptitle("H2 Ansatz Circuit (2 Qubits)", fontweight='bold')
fig.tight_layout()
savefig("01_ansatz_circuit_2q.png", fig)

# ==============================
# STEP 3: Initial Energy (2 qubit)
# ==============================
estimator = StatevectorEstimator()
np.random.seed(42)
params = np.random.rand(ansatz_2q.num_parameters)
job = estimator.run([(ansatz_2q, hamiltonian_2q, params)])
initial_energy = job.result()[0].data.evs
print("\n===== Initial Energy =====")
print(initial_energy)

# ==============================
# STEP 5: VQE for 2-qubit (FIXED)
# ==============================

def run_vqe_2q_single(seed):
    """Run one VQE trial with a given random seed, return (result, trace)."""
    np.random.seed(seed)
    init_pt = np.random.uniform(-np.pi, np.pi, ansatz_2q.num_parameters)
    trace = []

    def cb(eval_count, params, energy, std):
        trace.append(energy)

    opt = COBYLA(maxiter=500)
    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz_2q,
        optimizer=opt,
        callback=cb,
        initial_point=init_pt
    )
    res = vqe.compute_minimum_eigenvalue(hamiltonian_2q)
    return res, trace

print("\n===== Running 2-Qubit VQE (Multi-Start: 3 trials) =====")
best_result_2q = None
energy_values_2q = []
SEEDS = [42, 7, 123]

for trial, seed in enumerate(SEEDS):
    res, trace = run_vqe_2q_single(seed)
    e = res.eigenvalue.real
    print(f"  Trial {trial+1} (seed={seed}): Energy = {e:.6f} Ha")
    if best_result_2q is None or e < best_result_2q.eigenvalue.real:
        best_result_2q = res
        energy_values_2q = trace

result_2q = best_result_2q
print(f"\n  ✅ Best result selected: {result_2q.eigenvalue.real:.6f} Ha")

print("\n===== VQE Result (2 Qubits — FIXED) =====")
print("Ground State Energy:", result_2q.eigenvalue.real)

# Convergence plot for 2-qubit
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(energy_values_2q, marker='o', color='#5C4F4A')
ax.set_title("VQE Convergence (2 Qubits)", fontweight='bold')
ax.set_xlabel("Iteration", fontweight='bold')
ax.set_ylabel("Energy", fontweight='bold')
fig.tight_layout()
savefig("02_vqe_convergence_2q.png", fig)

# ==============================
# STEP 7: Detailed + Optimized Circuit (2 qubit)
# ==============================
detailed = ansatz_2q.decompose()
try:
    fig2 = detailed.draw(output='mpl', scale=1.5, fold=20)
    fig2.set_size_inches(16, 10)
    fig2.suptitle("Detailed H2 Circuit (2 Qubits)", fontweight='bold')
    savefig("03_detailed_circuit_2q.png", fig2)
except Exception as ex:
    print(f"  [WARN] Could not draw detailed circuit: {ex}")
    print(detailed.draw())

optimal_params_2q = result_2q.optimal_point
optimized_circuit_2q = ansatz_2q.assign_parameters(optimal_params_2q)
fig_opt = optimized_circuit_2q.draw(output='mpl', scale=1.5, fold=20)
fig_opt.set_size_inches(14, 8)
fig_opt.suptitle("Optimized Circuit - 2 Qubits (After COBYLA)", fontweight='bold')
savefig("04_optimized_circuit_2q.png", fig_opt)

final_energy_2q = result_2q.eigenvalue.real
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(energy_values_2q, marker='o', label='Energy', color='#6E1A37')
ax.scatter(len(energy_values_2q)-1, final_energy_2q, s=100, label='Optimal Solution')
ax.axhline(y=final_energy_2q, linestyle='--', label='Final Energy')
ax.set_title("Optimal Solution (2 Qubits)", fontweight='bold')
ax.set_xlabel("Iteration", fontweight='bold')
ax.set_ylabel("Energy", fontweight='bold')
ax.legend()
fig.tight_layout()
savefig("05_optimal_solution_2q.png", fig)


# ============================================================
# HELPER FUNCTIONS FOR STEPS 8-12
# ============================================================

def build_hamiltonian(n_qubits):
    if n_qubits == 2:
        return SparsePauliOp.from_list([
            ("ZZ", -1.05),
            ("IX", 0.39),
            ("XI", 0.39),
            ("II", -0.01)
        ])

    terms = []
    terms.append(("I" * n_qubits, -0.01 * n_qubits))

    for i in range(n_qubits - 1):
        pauli_str = list("I" * n_qubits)
        pauli_str[n_qubits - 1 - i] = "Z"
        pauli_str[n_qubits - 2 - i] = "Z"
        terms.append(("".join(pauli_str), -1.05))

    for i in range(n_qubits):
        pauli_str = list("I" * n_qubits)
        pauli_str[n_qubits - 1 - i] = "X"
        terms.append(("".join(pauli_str), 0.39))

    return SparsePauliOp.from_list(terms)


def classical_ground_state(hamiltonian):
    matrix = hamiltonian.to_matrix()
    eigenvalues = np.linalg.eigvalsh(matrix)
    return np.min(eigenvalues).real


def run_vqe_experiment(n_qubits, maxiter=200):
    print(f"\n{'='*50}")
    print(f"  RUNNING VQE EXPERIMENT: {n_qubits} QUBITS")
    print(f"{'='*50}")

    hamiltonian = build_hamiltonian(n_qubits)
    ansatz = TwoLocal(
        num_qubits=n_qubits,
        rotation_blocks='ry',
        entanglement_blocks='cx',
        reps=2
    )

    # ── Save Ansatz circuit for this qubit count ──────────────────────────
    print(f"  Saving Ansatz circuit for {n_qubits} qubits...")
    try:
        fig_ans = ansatz.draw(output='mpl', scale=1.2, fold=25)
        fig_ans.set_size_inches(16, max(6, n_qubits * 1.8))
        fig_ans.suptitle(f"Ansatz Circuit — {n_qubits} Qubits (TwoLocal, RY + CX, reps=2)",
                         fontweight='bold', fontsize=16)
        fig_ans.tight_layout()
        savefig(f"ansatz_circuit_{n_qubits}q.png", fig_ans)
    except Exception as ex:
        print(f"  [WARN] Could not draw ansatz circuit for {n_qubits}q: {ex}")
    # ─────────────────────────────────────────────────────────────────────

    energy_trace = []

    def callback(eval_count, params, energy, std):
        energy_trace.append(energy)

    est = StatevectorEstimator()
    optimizer = COBYLA(maxiter=maxiter)
    vqe = VQE(estimator=est, ansatz=ansatz, optimizer=optimizer, callback=callback)

    t_start = time.time()
    result = vqe.compute_minimum_eigenvalue(hamiltonian)
    t_end = time.time()

    vqe_energy = result.eigenvalue.real
    classical_energy = classical_ground_state(hamiltonian)
    runtime = t_end - t_start
    accuracy = abs(vqe_energy - classical_energy)
    convergence_iter = len(energy_trace)

    converge_at = convergence_iter
    for i, e in enumerate(energy_trace):
        if abs(e - vqe_energy) < 0.01 * abs(vqe_energy) + 1e-8:
            converge_at = i + 1
            break

    print(f"  VQE Energy      : {vqe_energy:.6f} Ha")
    print(f"  Classical Energy: {classical_energy:.6f} Ha")
    print(f"  Accuracy (|Δ|)  : {accuracy:.6f} Ha")
    print(f"  Iterations      : {convergence_iter}")
    print(f"  Converged at    : iteration {converge_at}")
    print(f"  Runtime         : {runtime:.2f} s")

    # ── Save Optimized (bound-parameter) Ansatz circuit ───────────────────
    try:
        opt_circuit = ansatz.assign_parameters(result.optimal_point)
        fig_opt_nq = opt_circuit.draw(output='mpl', scale=1.2, fold=25)
        fig_opt_nq.set_size_inches(16, max(6, n_qubits * 1.8))
        fig_opt_nq.suptitle(f"Optimized Ansatz Circuit — {n_qubits} Qubits (After COBYLA)",
                            fontweight='bold', fontsize=16)
        fig_opt_nq.tight_layout()
        savefig(f"ansatz_optimized_{n_qubits}q.png", fig_opt_nq)
    except Exception as ex:
        print(f"  [WARN] Could not draw optimized ansatz for {n_qubits}q: {ex}")
    # ─────────────────────────────────────────────────────────────────────

    return {
        "n_qubits": n_qubits,
        "vqe_energy": vqe_energy,
        "classical_energy": classical_energy,
        "accuracy": accuracy,
        "energy_trace": energy_trace,
        "runtime": runtime,
        "convergence_iter": convergence_iter,
        "converge_at": converge_at,
    }


# ============================================================
# STEP 8: Run Experiment for Qubits 3, 4, 5, 6
# ============================================================

all_results = {}

all_results[2] = {
    "n_qubits": 2,
    "vqe_energy": result_2q.eigenvalue.real,
    "classical_energy": classical_ground_state(hamiltonian_2q),
    "accuracy": abs(result_2q.eigenvalue.real - classical_ground_state(hamiltonian_2q)),
    "energy_trace": energy_values_2q,
    "runtime": None,
    "convergence_iter": len(energy_values_2q),
    "converge_at": len(energy_values_2q),
}

for nq in [3, 4, 5, 6]:
    all_results[nq] = run_vqe_experiment(nq, maxiter=200)

# ============================================================
# STEP 9: Performance plots — separate figure per qubit count
# ============================================================

colors  = {2: '#6E1A37', 3: '#1A3D6E', 4: '#1A6E3D', 5: '#6E5A1A', 6: '#5A1A6E'}
markers = {2: 'o',       3: 's',       4: '^',       5: 'D',       6: 'P'}

for nq in [3, 4, 5, 6]:
    res = all_results[nq]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Performance Metrics — {nq} Qubits", fontweight='bold', fontsize=20)

    axes[0].plot(res["energy_trace"], marker=markers[nq], color=colors[nq],
                 markevery=max(1, len(res["energy_trace"])//20), linewidth=2)
    axes[0].axhline(y=res["classical_energy"], linestyle='--', color='black', linewidth=1.5,
                    label=f'Classical: {res["classical_energy"]:.4f} Ha')
    axes[0].axhline(y=res["vqe_energy"], linestyle=':', color=colors[nq], linewidth=1.5,
                    label=f'VQE Final: {res["vqe_energy"]:.4f} Ha')
    axes[0].set_title(f"Energy Convergence ({nq} Qubits)", fontweight='bold')
    axes[0].set_xlabel("Iteration", fontweight='bold')
    axes[0].set_ylabel("Energy (Ha)", fontweight='bold')
    axes[0].legend(fontsize=14)
    axes[0].grid(True, alpha=0.3)

    bar_labels = ['VQE Energy', 'Classical Energy']
    bar_values = [res["vqe_energy"], res["classical_energy"]]
    bar_colors = [colors[nq], '#333333']
    bars = axes[1].bar(bar_labels, bar_values, color=bar_colors, width=0.4, edgecolor='black')
    for bar, val in zip(bars, bar_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    axes[1].set_title(f"VQE vs Classical ({nq} Qubits)", fontweight='bold')
    axes[1].set_ylabel("Energy (Ha)", fontweight='bold')
    axes[1].set_ylim(min(bar_values) * 1.05, max(bar_values) * 0.95)
    axes[1].grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    savefig(f"06_performance_{nq}q.png", fig)

# ============================================================
# STEP 10: Quantum vs Classical — separate figure per qubit
# ============================================================

for nq in [3, 4, 5, 6]:
    res = all_results[nq]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Quantum vs Classical Comparison — {nq} Qubits", fontweight='bold', fontsize=20)

    methods  = ['VQE (Quantum)', 'NumPy Exact\n(Classical)']
    energies = [res["vqe_energy"], res["classical_energy"]]
    bcolors  = [colors[nq], '#2C2C2C']

    bars = axes[0].bar(methods, energies, color=bcolors, width=0.5, edgecolor='black', linewidth=1.2)
    for bar, val in zip(bars, energies):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + (max(energies) - min(energies)) * 0.01,
                     f'{val:.5f} Ha', ha='center', va='bottom', fontsize=14, fontweight='bold')
    axes[0].set_title(f"Energy: Quantum vs Classical ({nq}q)", fontweight='bold')
    axes[0].set_ylabel("Ground State Energy (Ha)", fontweight='bold')
    axes[0].set_ylim(min(energies) * 1.1, max(energies) * 0.9)
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(['Energy Error |VQE - Classical|'], [res["accuracy"]],
                color='#C0392B', width=0.4, edgecolor='black', linewidth=1.2)
    axes[1].text(0, res["accuracy"] + res["accuracy"] * 0.05,
                 f'{res["accuracy"]:.6f} Ha', ha='center', va='bottom', fontsize=15, fontweight='bold')
    axes[1].set_title(f"Absolute Error ({nq} Qubits)", fontweight='bold')
    axes[1].set_ylabel("| VQE - Classical | (Ha)", fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    savefig(f"07_quantum_vs_classical_{nq}q.png", fig)

# ============================================================
# STEP 11: Convergence plots — separate figure per qubit
# ============================================================

for nq in [3, 4, 5, 6]:
    res = all_results[nq]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"VQE Convergence Analysis — {nq} Qubits", fontweight='bold', fontsize=20)

    trace = res["energy_trace"]
    axes[0].plot(trace, color=colors[nq], linewidth=2,
                 marker=markers[nq], markevery=max(1, len(trace)//20), markersize=6)
    axes[0].scatter(len(trace)-1, res["vqe_energy"], s=200, color='red',
                    zorder=5, label=f'Optimal: {res["vqe_energy"]:.4f} Ha')
    axes[0].axhline(y=res["classical_energy"], linestyle='--', color='black', linewidth=1.5,
                    label=f'Exact: {res["classical_energy"]:.4f} Ha')
    axes[0].set_title(f"Energy vs Iterations ({nq}q)", fontweight='bold')
    axes[0].set_xlabel("Iteration", fontweight='bold')
    axes[0].set_ylabel("Energy (Ha)", fontweight='bold')
    axes[0].legend(fontsize=13)
    axes[0].grid(True, alpha=0.3)

    accuracy_over_iters = [abs(e - res["classical_energy"]) for e in trace]
    axes[1].semilogy(accuracy_over_iters, color=colors[nq], linewidth=2)
    axes[1].set_title(f"Accuracy vs Iterations ({nq}q)", fontweight='bold')
    axes[1].set_xlabel("Iteration", fontweight='bold')
    axes[1].set_ylabel("|Energy Error| (Ha) — log scale", fontweight='bold')
    axes[1].grid(True, alpha=0.3, which='both')
    axes[1].axhline(y=1e-3, linestyle=':', color='gray', label='1e-3 threshold')
    axes[1].legend(fontsize=13)

    fig.tight_layout()
    savefig(f"08_convergence_analysis_{nq}q.png", fig)

# ============================================================
# STEP 12: Final Summary — all qubits combined
# ============================================================

all_nq       = sorted(all_results.keys())
vqe_ens      = [all_results[nq]["vqe_energy"]      for nq in all_nq]
cls_ens      = [all_results[nq]["classical_energy"] for nq in all_nq]
accs         = [all_results[nq]["accuracy"]         for nq in all_nq]
iters        = [all_results[nq]["convergence_iter"] for nq in all_nq]
runtime_plot = [all_results[nq]["runtime"] or 0     for nq in all_nq]

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Final Summary: VQE vs Classical (Qubits 2–6)", fontweight='bold', fontsize=22)

col_list = [colors[nq] for nq in all_nq]
x = np.arange(len(all_nq))
width = 0.35

# Energy comparison
axes[0, 0].bar(x - width/2, vqe_ens, width, label='VQE (Quantum)',
               color=col_list, edgecolor='black')
axes[0, 0].bar(x + width/2, cls_ens, width, label='Classical (NumPy)',
               color='#444444', edgecolor='black')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels([f'{nq}q' for nq in all_nq])
axes[0, 0].set_title("Ground State Energy: VQE vs Classical", fontweight='bold')
axes[0, 0].set_ylabel("Energy (Ha)", fontweight='bold')
axes[0, 0].legend(fontsize=14)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Accuracy
bars3 = axes[0, 1].bar([f'{nq}q' for nq in all_nq], accs, color=col_list, edgecolor='black')
for bar, val in zip(bars3, accs):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(accs)*0.02,
                    f'{val:.4f}', ha='center', fontsize=12, fontweight='bold')
axes[0, 1].set_title("Energy Accuracy |VQE - Classical|", fontweight='bold')
axes[0, 1].set_ylabel("Absolute Error (Ha)", fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Convergence speed
bars4 = axes[1, 0].bar([f'{nq}q' for nq in all_nq], iters, color=col_list, edgecolor='black')
for bar, val in zip(bars4, iters):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(iters)*0.01,
                    str(val), ha='center', fontsize=12, fontweight='bold')
axes[1, 0].set_title("Total Iterations to Converge", fontweight='bold')
axes[1, 0].set_ylabel("Iterations", fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Runtime
bars5 = axes[1, 1].bar([f'{nq}q' for nq in all_nq], runtime_plot, color=col_list, edgecolor='black')
for bar, val in zip(bars5, runtime_plot):
    if val > 0:
        axes[1, 1].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(max(runtime_plot), 0.1)*0.01,
                        f'{val:.1f}s', ha='center', fontsize=12, fontweight='bold')
axes[1, 1].set_title("Runtime per Experiment", fontweight='bold')
axes[1, 1].set_ylabel("Time (seconds)", fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

fig.tight_layout()
savefig("09_final_summary.png", fig)

# ============================================================
# Print Final Summary Table
# ============================================================
print("\n" + "="*75)
print(f"{'FINAL SUMMARY TABLE':^75}")
print("="*75)
print(f"{'Qubits':<10} {'VQE Energy':>14} {'Classical':>14} {'|Error|':>12} {'Iters':>8} {'Time(s)':>10}")
print("-"*75)
for nq in all_nq:
    res = all_results[nq]
    rt = f"{res['runtime']:.2f}" if res['runtime'] else "N/A"
    print(f"{nq:<10} {res['vqe_energy']:>14.6f} {res['classical_energy']:>14.6f} "
          f"{res['accuracy']:>12.6f} {res['convergence_iter']:>8} {rt:>10}")
print("="*75)

print(f"\n✅ All plots saved to: ./{OUTPUT_DIR}/")
print(f"   Total figures: {len(os.listdir(OUTPUT_DIR))}")