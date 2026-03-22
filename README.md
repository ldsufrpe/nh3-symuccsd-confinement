# SymUCCSD DLA Confinement — NH₃/STO-3G

Numerical validation code for:

> **Lie-algebraic incompleteness of symmetry-adapted VQE for non-Abelian molecular point groups**
> Leon D. da Silva & Marcelo P. Santos (2025)
> [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

## Overview

This repository contains the simulation code that reproduces the numerical results in Table 1 of the paper. The code demonstrates the **Dynamical Lie Algebra (DLA) confinement** of the SymUCCSD ansatz for ammonia (NH₃) under the non-Abelian point group C₃ᵥ.

The SymUCCSD method filters excitation operators using the Abelian subgroup Cₛ ≤ C₃ᵥ. For NH₃/STO-3G (16 qubits, 10 electrons), this reduces the UCCSD pool from 135 to 75 spin-complemented parameters. However, the retained generators commute pairwise, confining the reachable manifold to a torus T² — a measure-zero submanifold of the full equivariant unitary group U(2). The VQE optimizer converges to an energy plateau ΔE ≈ 21.8 mHa above the FCI reference, confirming the structural confinement predicted by Theorem 5.1.

## Results

| Method | Parameters | ΔE (mHa) |
|--------|-----------|-----------|
| HF | — | 65.9 |
| SymUCCSD (Cₛ) | 75 | 21.8 |
| UCCSD (full) | 135 | 0.1 |
| FCI | — | 0.0 |

## Requirements

```
pip install numpy scipy pyscf openfermion
```

Tested with Python 3.10–3.12, PySCF 2.5+, OpenFermion 1.6+.

## Usage

Run the SymUCCSD confinement experiment:

```bash
python nh3_symuccsd_confinement.py
```

Also run the full UCCSD for comparison (slower, ~135 parameters):

```bash
python nh3_symuccsd_confinement.py --run-uccsd
```

Pipeline test without optimization:

```bash
python nh3_symuccsd_confinement.py --dry-run
```

Results are saved as a JSON file with full convergence history, suitable for plotting.

## Protocol

Following He et al. ([arXiv:2512.21087](https://arxiv.org/abs/2512.21087)):

- **Molecule:** NH₃ at equilibrium geometry (STO-3G basis)
- **Qubits:** 16 (8 spatial MOs, Jordan-Wigner mapping, no frozen core)
- **Electrons:** 10
- **Excitations:** Spin-complemented UCCSD with `i ≤ j`, `a ≤ b` convention for opposite-spin doubles
- **SymUCCSD filter:** Cₛ subgroup irrep labels via PySCF (`symmetry='Cs'`)
- **Simulation:** Exact statevector via `scipy.sparse.linalg.expm_multiply`
- **Optimizer:** BFGS with `gtol = 1e-4`

## Output format

The script saves a JSON file containing:

- Molecular energies (HF, CCSD, FCI)
- VQE results per method (energy, parameter count, function evaluations)
- Full convergence history (energy at every function evaluation)

Example:

```json
{
  "molecule": "NH3",
  "basis": "sto-3g",
  "n_qubits": 16,
  "e_fci": -55.5204705385,
  "results": {
    "SymUCCSD": {
      "energy_ha": -55.4986434886,
      "delta_e_mha": 21.827,
      "n_params": 75,
      "n_function_evals": 3801
    }
  },
  "histories": {
    "SymUCCSD": [
      {"nfev": 1, "energy": -55.4545, "best_energy": -55.4545, "elapsed_s": 0.04},
      ...
    ]
  }
}
```

## Citation

```bibtex
@article{daSilva2025,
  author  = {da Silva, Leon D. and Santos, Marcelo P.},
  title   = {Lie-algebraic incompleteness of symmetry-adapted {VQE} for non-{A}belian molecular point groups},
  year    = {2025},
  eprint  = {XXXX.XXXXX},
  archiveprefix = {arXiv},
  primaryclass  = {quant-ph}
}
```

## License

MIT
