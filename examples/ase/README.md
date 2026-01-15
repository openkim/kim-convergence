# ASE Examples for kim-convergence

This directory contains examples demonstrating how to use kim-convergence with
the Atomic Simulation Environment (ASE) for convergence-controlled equilibration.

## Prerequisites

Install ASE:

```bash
pip install ase
```

## Examples

### `example_equilibration.py`

A comprehensive example showing:

1. **Basic equilibration**: Run Langevin dynamics on a Cu system until temperature
   equilibrates to the specified accuracy.

2. **Sample interval**: How to sample less frequently for expensive calculators.

3. **Custom property extractors**: How to define and use your own property
   extractors (e.g., maximum force).

4. **Energy-based equilibration**: Monitor potential energy for convergence.

Run:

```bash
python example_equilibration.py
```

## API Usage

### Basic Usage

```python
from kim_convergence.ase import ASESampler, run_ase_equilibration

# Create sampler
sampler = ASESampler(dyn, property_name="temperature")

# Run equilibration
result = run_ase_equilibration(
    sampler,
    initial_run_length=1000,
    maximum_run_length=100000,
    relative_accuracy=0.05,
)

if result["converged"]:
    print(f"Equilibrated in {result['total_run_length']} samples")
    print(f"Mean value: {result['mean']}")
```

### Using `sample_interval` for Expensive Calculators

For expensive calculators (e.g., neural network potentials), sample less frequently:

```python
# Sample every 10 MD steps
sampler = ASESampler(dyn, property_name="energy", sample_interval=10)

result = run_ase_equilibration(
    sampler,
    initial_run_length=100,   # 100 samples = 1000 MD steps
    maximum_run_length=5000,  # 5000 samples = 50000 MD steps
    relative_accuracy=0.05,
)
```

### Available Properties

Built-in property extractors:

- `"energy"` or `"potential_energy"`: Potential energy (eV)
- `"kinetic_energy"`: Kinetic energy (eV)
- `"total_energy"`: Total energy (eV)
- `"temperature"`: Kinetic temperature (K)
- `"volume"`: Cell volume (Å³)
- `"pressure"`: Hydrostatic pressure (eV/Å³)
- `"density"`: Mass density (g/cm³)

### Custom Properties

You can define custom property extractors:

```python
def get_max_displacement(atoms):
    """Example: track maximum atomic displacement."""
    return float(np.max(np.linalg.norm(atoms.positions - initial_positions, axis=1)))

sampler = ASESampler(
    dyn,
    property_name="max_displacement",
    extractors={"max_displacement": get_max_displacement},
)

result = run_ase_equilibration(sampler, relative_accuracy=0.1)
```

## Return Value

The `run_ase_equilibration` function returns the kim-convergence result dictionary.
Key fields include:

| Key | Description |
|-----|-------------|
| `converged` | Whether convergence was achieved (bool) |
| `total_run_length` | Total samples collected (int) |
| `equilibration_step` | Sample where equilibration was detected (int) |
| `mean` | Estimated mean of monitored property (float) |
| `standard_deviation` | Standard deviation (float) |
| `effective_sample_size` | Number of independent samples (float) |
| `upper_confidence_limit` | Upper confidence limit (float) |
| `relative_half_width` | Achieved relative accuracy (float) |

See `kim_convergence.run_length_control` documentation for all available fields.
