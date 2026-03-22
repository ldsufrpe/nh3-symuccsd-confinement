#!/usr/bin/env python3
"""
nh3_symuccsd_confinement.py
===========================
Numerical validation for:
  "Lie-algebraic incompleteness of symmetry-adapted VQE
   for non-Abelian molecular point groups"
  L. D. da Silva & M. P. Santos (2025)

Reproduces Table 1: SymUCCSD DLA confinement on NH3/STO-3G.

Protocol (following He et al., arXiv:2512.21087):
  - 16 qubits (8 spatial MOs, no frozen core)
  - 10 electrons
  - STO-3G basis
  - Jordan-Wigner mapping
  - Spin-complemented UCCSD excitations (i<=j, a<=b for OS doubles)
  - SymUCCSD: Cs subgroup filter (PySCF symmetry='Cs')
  - Statevector simulation with BFGS optimizer

Expected results:
  SymUCCSD (Cs, 75 params):  ΔE ≈ 21.8 mHa  (DLA confinement)
  UCCSD   (full, 135 params): ΔE ≈ 0.1 mHa   (no confinement)

Requirements:
  pip install numpy scipy pyscf openfermion

Usage:
  python nh3_symuccsd_confinement.py                  # SymUCCSD only (fast)
  python nh3_symuccsd_confinement.py --run-uccsd      # also run UCCSD (slow)
  python nh3_symuccsd_confinement.py --dry-run        # pipeline test, no optimization
"""

import numpy as np
import json
import time
import argparse
from datetime import datetime
from itertools import (combinations,
                       combinations_with_replacement as cwr)
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import minimize


def log(msg):
    print(msg, flush=True)


# ═══════════════════════════════════════════════════════════════
# 1. MOLECULAR SETUP
# ═══════════════════════════════════════════════════════════════

def run_pyscf():
    """
    NH3/STO-3G: RHF with Cs symmetry, FCI in full space.
    Returns integrals, energies, and Cs irrep labels.
    """
    from pyscf import gto, scf, cc, fci, ao2mo

    log("Phase 0: PySCF (16 qubits, no frozen core) ...")

    mol = gto.M(
        atom="""
            N   0.0000   0.0000   0.1173
            H   0.0000   0.9377  -0.2738
            H   0.8126  -0.4689  -0.2738
            H  -0.8126  -0.4689  -0.2738
        """,
        basis='sto-3g', symmetry='Cs', unit='Angstrom', verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    mycc = cc.CCSD(mf)
    mycc.frozen = None
    mycc.kernel()

    cisolver = fci.FCI(mf)
    cisolver.verbose = 0
    e_fci, _ = cisolver.kernel()

    nmo = mol.nao
    mo_coeff = mf.mo_coeff
    h1 = mo_coeff.T @ mf.get_hcore() @ mo_coeff
    h2 = ao2mo.kernel(mol, mo_coeff)
    h2 = ao2mo.restore(1, h2, nmo)

    orbsym = [int(x) for x in mf.get_orbsym()]
    nocc = mol.nelectron // 2

    log(f"  E(HF)   = {mf.e_tot:.10f} Ha")
    log(f"  E(CCSD) = {mycc.e_tot:.10f} Ha")
    log(f"  E(FCI)  = {e_fci:.10f} Ha")
    log(f"  Qubits  = {2 * nmo}, Electrons = {mol.nelectron}")
    log(f"  Cs irreps = {orbsym}")

    return dict(
        e_hf=mf.e_tot, e_ccsd=mycc.e_tot, e_fci=e_fci,
        e_nuc=mol.energy_nuc(), h1=h1, h2=h2,
        n_spatial=nmo, n_elec=mol.nelectron, nocc=nocc,
        orbsym=orbsym,
    )


# ═══════════════════════════════════════════════════════════════
# 2. HAMILTONIAN (Jordan-Wigner)
# ═══════════════════════════════════════════════════════════════

def build_hamiltonian(mol_data):
    """Build sparse qubit Hamiltonian and HF state vector."""
    from openfermion import FermionOperator, jordan_wigner, get_sparse_operator

    log("Phase 1: Hamiltonian ...")
    t0 = time.time()

    N = mol_data['n_spatial']
    h1, h2 = mol_data['h1'], mol_data['h2']
    e_nuc = mol_data['e_nuc']
    n_qubits = 2 * N

    H_ferm = FermionOperator((), e_nuc)
    for p in range(N):
        for q in range(N):
            if abs(h1[p, q]) < 1e-14:
                continue
            H_ferm += FermionOperator(((p, 1), (q, 0)), h1[p, q])
            H_ferm += FermionOperator(((p+N, 1), (q+N, 0)), h1[p, q])
    for p in range(N):
        for q in range(N):
            for r in range(N):
                for s in range(N):
                    c = 0.5 * h2[p, q, r, s]
                    if abs(c) < 1e-14:
                        continue
                    H_ferm += FermionOperator(((p,1),(r,1),(s,0),(q,0)), c)
                    H_ferm += FermionOperator(((p,1),(r+N,1),(s+N,0),(q,0)), c)
                    H_ferm += FermionOperator(((p+N,1),(r,1),(s,0),(q+N,0)), c)
                    H_ferm += FermionOperator(((p+N,1),(r+N,1),(s+N,0),(q+N,0)), c)

    H_sparse = get_sparse_operator(jordan_wigner(H_ferm), n_qubits=n_qubits)

    # HF state
    nocc = mol_data['nocc']
    occ = list(range(nocc)) + [k + N for k in range(nocc)]
    idx = 0
    for k in occ:
        idx |= (1 << (n_qubits - 1 - k))
    hf = np.zeros(2**n_qubits, dtype=complex)
    hf[idx] = 1.0

    e_check = np.real(hf.conj() @ H_sparse @ hf)
    assert abs(e_check - mol_data['e_hf']) < 1e-6, \
        f"HF energy mismatch: {e_check} vs {mol_data['e_hf']}"

    dt = time.time() - t0
    log(f"  {n_qubits} qubits, dim = {2**n_qubits}, built in {dt:.1f}s")

    return H_sparse, hf, n_qubits


# ═══════════════════════════════════════════════════════════════
# 3. POOL CONSTRUCTION
# ═══════════════════════════════════════════════════════════════

def _to_sparse(ferm_op, n_qubits):
    """Convert T to sparse anti-Hermitian generator (T - T†)."""
    from openfermion import (hermitian_conjugated, normal_ordered,
                             jordan_wigner, get_sparse_operator)
    A = ferm_op - hermitian_conjugated(ferm_op)
    A = normal_ordered(A)
    A.compress()
    if len(A.terms) == 0:
        return None
    return get_sparse_operator(jordan_wigner(A), n_qubits=n_qubits)


def build_pool(mol_data, n_qubits, method='symuccsd'):
    """
    Build spin-complemented generator pool.

    method='symuccsd': Cs-filtered (Abelian subgroup filter)
    method='uccsd':    unfiltered (full UCCSD)
    """
    from openfermion import FermionOperator

    N = mol_data['n_spatial']
    nocc = mol_data['nocc']
    orbsym = mol_data['orbsym']
    occ = list(range(nocc))
    vir = list(range(nocc, N))

    log(f"  Building {method.upper()} pool ...")
    t0 = time.time()

    generators = []
    n_singles, n_ss, n_os = 0, 0, 0

    # Singles
    for i in occ:
        for a in vir:
            if method == 'symuccsd' and orbsym[i] != orbsym[a]:
                continue
            T = FermionOperator(((a, 1), (i, 0)))
            T += FermionOperator(((a+N, 1), (i+N, 0)))
            sp = _to_sparse(T, n_qubits)
            if sp is not None:
                generators.append(sp)
                n_singles += 1

    # Same-spin doubles
    for i, j in combinations(occ, 2):
        for a, b in combinations(vir, 2):
            if method == 'symuccsd':
                if (orbsym[i] ^ orbsym[j] ^ orbsym[a] ^ orbsym[b]) != 0:
                    continue
            T = FermionOperator(((a, 1), (b, 1), (j, 0), (i, 0)))
            T += FermionOperator(((a+N, 1), (b+N, 1), (j+N, 0), (i+N, 0)))
            sp = _to_sparse(T, n_qubits)
            if sp is not None:
                generators.append(sp)
                n_ss += 1

    # Opposite-spin doubles (cwr convention: i<=j, a<=b)
    for i, j in cwr(occ, 2):
        for a, b in cwr(vir, 2):
            if method == 'symuccsd':
                if (orbsym[i] ^ orbsym[j] ^ orbsym[a] ^ orbsym[b]) != 0:
                    continue
            T = FermionOperator(((a, 1), (b+N, 1), (j+N, 0), (i, 0)))
            sp = _to_sparse(T, n_qubits)
            if sp is not None:
                generators.append(sp)
                n_os += 1

    dt = time.time() - t0
    log(f"  {method.upper()}: {len(generators)} generators "
        f"(S={n_singles}, D_ss={n_ss}, D_os={n_os}) in {dt:.1f}s")

    return generators


# ═══════════════════════════════════════════════════════════════
# 4. VQE ENGINE
# ═══════════════════════════════════════════════════════════════

def run_vqe(generators, H_sparse, hf_state, e_fci, label,
            maxiter=15000, dry_run=False):
    """Statevector VQE with BFGS optimizer."""
    n_params = len(generators)

    if dry_run:
        log(f"\n  VQE {label} — DRY RUN ({n_params} params)")
        psi = hf_state.copy()
        e0 = np.real(psi.conj() @ H_sparse @ psi)
        t0 = time.time()
        for k in range(min(3, n_params)):
            psi = expm_multiply(0.001 * generators[k], psi)
        dt = time.time() - t0
        log(f"  Pipeline OK. Est. {dt / max(1, min(3, n_params)) * n_params * maxiter / 60:.0f} min")
        return e0, 0, []

    log(f"\n  VQE {label} ({n_params} params, BFGS, maxiter={maxiter})")

    nfev = [0]
    best_e = [np.inf]
    t0 = time.time()
    history = []

    def cost(theta):
        psi = hf_state.copy()
        for k in range(n_params):
            if abs(theta[k]) > 1e-15:
                psi = expm_multiply(theta[k] * generators[k], psi)
        energy = np.real(psi.conj() @ H_sparse @ psi)
        nfev[0] += 1
        if energy < best_e[0]:
            best_e[0] = energy
        history.append((nfev[0], float(energy), float(best_e[0]),
                        round(time.time() - t0, 2)))
        if nfev[0] % 500 == 0 or nfev[0] == 1:
            dt = time.time() - t0
            de = (best_e[0] - e_fci) * 1000
            print(f"    [{label}] nfev={nfev[0]:>6d}  E={best_e[0]:.8f}  "
                  f"ΔE={de:>8.3f} mHa  (t={dt:.0f}s)", flush=True)
        return energy

    # Timing estimate
    t_test = time.time()
    _ = cost(np.zeros(n_params))
    dt_single = time.time() - t_test
    log(f"  Single eval: {dt_single:.3f}s → est. max "
        f"{dt_single * maxiter / 60:.1f} min")
    history.clear()
    nfev[0] = 0
    best_e[0] = np.inf

    res = minimize(cost, np.zeros(n_params), method='BFGS',
                   options={'maxiter': maxiter, 'gtol': 1e-4, 'disp': False})

    de = (res.fun - e_fci) * 1000
    dt = time.time() - t0
    log(f"  ✓ {label}: E={res.fun:.10f}  ΔE={de:.3f} mHa  "
        f"nfev={nfev[0]}  t={dt:.0f}s  success={res.success}")

    return res.fun, nfev[0], history


# ═══════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce SymUCCSD DLA confinement on NH3/STO-3G "
                    "(da Silva & Santos, 2025)")
    parser.add_argument('--run-uccsd', action='store_true',
                        help='Also run full UCCSD (135 params, slow)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Pipeline test only, no optimization')
    args = parser.parse_args()

    log("=" * 65)
    log("  SymUCCSD DLA Confinement — NH₃ / STO-3G")
    log("  da Silva & Santos (2025)")
    log("  16 qubits | no frozen core | BFGS optimizer")
    log("=" * 65)

    mol_data = run_pyscf()
    H_sparse, hf_state, n_qubits = build_hamiltonian(mol_data)
    e_fci = mol_data['e_fci']

    results = {}
    histories = {}

    # SymUCCSD
    sym_gens = build_pool(mol_data, n_qubits, method='symuccsd')
    e, nfev, hist = run_vqe(sym_gens, H_sparse, hf_state, e_fci,
                             'SymUCCSD', maxiter=15000, dry_run=args.dry_run)
    results['SymUCCSD'] = (e, nfev, len(sym_gens))
    histories['SymUCCSD'] = hist
    del sym_gens

    # UCCSD (optional)
    if args.run_uccsd:
        uccsd_gens = build_pool(mol_data, n_qubits, method='uccsd')
        e, nfev, hist = run_vqe(uccsd_gens, H_sparse, hf_state, e_fci,
                                 'UCCSD', maxiter=20000, dry_run=args.dry_run)
        results['UCCSD'] = (e, nfev, len(uccsd_gens))
        histories['UCCSD'] = hist
        del uccsd_gens

    # Results table
    log("\n" + "=" * 65)
    log("  RESULTS — NH₃/STO-3G, 16 qubits, no frozen core")
    if args.dry_run:
        log("  ⚠  DRY RUN — energies are NOT converged")
    log("=" * 65)
    log(f"{'Method':<20} {'Params':>7} {'Energy (Ha)':>18} {'ΔE (mHa)':>10}")
    log("-" * 59)
    log(f"{'HF':<20} {'—':>7} {mol_data['e_hf']:>18.10f} "
        f"{(mol_data['e_hf']-e_fci)*1000:>10.3f}")

    for label in ['SymUCCSD', 'UCCSD']:
        if label in results:
            e_val, _, np_ = results[label]
            log(f"{label:<20} {np_:>7d} {e_val:>18.10f} "
                f"{(e_val-e_fci)*1000:>10.3f}")

    if 'UCCSD' not in results:
        log(f"{'UCCSD (He et al.)':<20} {'135':>7} {'—':>18} {'0.109':>10}")

    log(f"{'FCI':<20} {'—':>7} {e_fci:>18.10f} {'0.000':>10}")
    log("-" * 59)

    # Save JSON
    json_results = {}
    for label, (e_val, nfev_val, np_val) in results.items():
        json_results[label] = {
            "energy_ha": e_val,
            "delta_e_mha": (e_val - e_fci) * 1000,
            "n_params": np_val,
            "n_function_evals": nfev_val,
        }

    json_histories = {}
    for label, hist in histories.items():
        json_histories[label] = [
            {"nfev": h[0], "energy": h[1], "best_energy": h[2],
             "elapsed_s": h[3]}
            for h in hist
        ]

    json_data = {
        "paper": "da Silva & Santos (2025), Lie-algebraic incompleteness "
                 "of symmetry-adapted VQE",
        "molecule": "NH3",
        "basis": "sto-3g",
        "n_qubits": n_qubits,
        "n_electrons": mol_data['n_elec'],
        "frozen_core": False,
        "optimizer": "BFGS",
        "gtol": 1e-4,
        "e_hf": mol_data['e_hf'],
        "e_ccsd": mol_data['e_ccsd'],
        "e_fci": e_fci,
        "results": json_results,
        "histories": json_histories,
        "timestamp": datetime.now().isoformat(),
        "dry_run": args.dry_run,
    }

    fname = f"nh3_confinement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, 'w') as f:
        json.dump(json_data, f, indent=2)
    log(f"\n  💾 Saved: {fname}")
    log("=" * 65)


if __name__ == '__main__':
    main()
