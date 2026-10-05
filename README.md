# Mapping the Safe Operating Envelope of a PID-Controlled Exothermic CSTR

Computational files for the study: *Mapping the Safe Operating Envelope of a PID-Controlled Exothermic CSTR under Simultaneous Kinetic Uncertainty and Disturbance*.

Author: Bihan Chakraborty  
Affiliation: Auxilium Convent School, Barasat

The study examines a PID-controlled exothermic continuous stirred-tank reactor (CSTR) under changes in feed temperature, feed anhydride concentration, coolant supply temperature, and selected kinetic parameters. The analysis combines DWSIM process simulation with Python-based dynamic simulation, deterministic Safe Operating Envelope (SOE) mapping, Monte Carlo probabilistic safety mapping, and heat-of-reaction sensitivity analysis.

The repository contains the Python implementation, the original manuscript, and the DWSIM project archive. These files are retained in their supplied form.

## Run order

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the computational script from the repository root:

```bash
python code/simulation.py
```

The script performs the steady-state calculations and checks, PID-controlled dynamic scenarios, SOE classification, the 729-point deterministic sweep, the 500-sample Monte Carlo analysis, and the heat-of-reaction sensitivity calculations.

The supplied source writes some outputs to Windows-specific paths under `C:\Users\bihan\`. Those paths may need to be changed on another machine. The archived source has not been changed for portability.

The DWSIM project is stored under `dwsim/` as supplied.

## What the model does

The reaction is:

```text
(CH3CO)2O + H2O -> 2 CH3COOH
```

The Python model uses the autocatalytic rate expression:

```text
r = Ca * Cw * (k_nc + k_cat' * [H3O+] / [H2O])
```

with:

```text
[H3O+] = sqrt(Ka * Cp)
[H2O] = 55.5 kmol/m³
```

The kinetic parameters in the source are:

| Parameter | Value |
|---|---:|
| `k_nc(313.15 K)` | `2.50e-5 m³/(kmol·s)` |
| `k_cat(313.15 K)` | `9.83e-2 m³/(kmol·s)` |
| `Ea_nc` | `46,000 J/mol` |
| `Ea_cat` | `29,700 J/mol` |
| `Ka` | `1.8e-5 kmol/m³` |
| `Tref` | `313.15 K` |

The reactor model uses a state vector of `[Ca, Cw, Cp, T]`. Cooling is represented by:

```text
Q_cool = UA * (T - Tc)
```

Nominal model values are:

| Quantity | Value |
|---|---:|
| Reactor volume, `V` | `1.0 m³` |
| Volumetric flow, `Q_vol` | `0.00317316 m³/s` |
| Feed temperature, `T0` | `303.0 K` |
| Feed anhydride concentration, `Ca0` | `5.0423 kmol/m³` |
| Coolant temperature, `Tc` | `298.0 K` |
| `UA` | approximately `1.976 kW/K` |
| Mixture heat capacity, `Cp_mix` | `97.60 kJ/(kmol·K)` |

The DWSIM project supplied with the archive identifies build version 9.0.5.0.

## Safety criterion and control

The baseline safety limit is:

```text
Tmax = Ts + f_safety * ΔTad
```

with:

```text
Ts = 330.0 K
f_safety = 0.75
ΔTad = 82.44 K
Tmax = 391.83 K
```

A case is classified as unsafe when the reactor temperature remains above `Tmax` for longer than the defined recovery window.

The controller is reverse acting. Higher reactor temperature causes a lower coolant temperature and therefore more cooling.

```text
Kp = 4.3
Ki = 0.05
Kd = 50.0
dt = 2.0 s
Coolant bounds = 278–320 K
```

Dynamic integration uses fourth-order Runge-Kutta. For SOE classification, the simulation duration is `10 × tau`, where `tau = V / Q_vol`, and the recovery window is `5 × tau`.

The code first evaluates the nominal steady state. It also performs a coupled nonlinear steady-state solve and checks multiple initial temperature guesses for possible multiple steady states, with residual and physical-validity checks applied to the verified test.

## Deterministic SOE mapping

The deterministic study evaluates `9 × 9 × 9 = 729` operating points:

| Variable | Range |
|---|---|
| Feed temperature, `T0` | 293–350 K |
| Feed anhydride concentration, `Ca0` | 3.8–6.3 kmol/m³ |
| Coolant supply temperature, `Tc_supply` | 288–320 K |

Each point is initialized at its coupled steady state, simulated under PID control, and classified as `SAFE` or `UNSAFE` from the temperature-excursion criterion.

The reported baseline result is:

- 594 SAFE points (81.5%)
- 135 UNSAFE points (18.5%)

The manuscript reports that the whole tested `T0` and `Tc_supply` range is SAFE at `Ca0 = 3.8 kmol/m³`. Near the nominal concentration of `5.05 kmol/m³`, the unsafe region appears at about `T0 >= 343 K`. At `Ca0 = 6.3 kmol/m³`, the unsafe region begins at about `T0 >= 328 K`, with coolant temperature affecting points near the boundary.

Feed temperature and feed concentration are identified as the main factors defining the safety boundary.

## Monte Carlo probabilistic mapping

The probabilistic study uses 500 samples with NumPy seed `42`.

| Variable | Distribution | Clipping bounds |
|---|---|---|
| `T0` | Normal(303.0, 15.0) K | 280–360 K |
| `Ca0` | Normal(5.0423, 0.8) kmol/m³ | 1.0 to `C_total` |
| `Tc_supply` | Normal(298.0, 8.0) K | 278–320 K |

The reported baseline result is:

- 491 SAFE
- 9 UNSAFE
- estimated safe probability: 98.2%

The unsafe samples are concentrated in the higher-feed-temperature part of the sampled space and under less favorable coolant conditions.

The deterministic and Monte Carlo percentages answer different questions. The deterministic grid gives equal weight to all tested combinations, while Monte Carlo samples are concentrated around the nominal operating point. The two percentages therefore should not be treated as estimates of the same quantity.

## Heat-of-reaction sensitivity

The main calculation uses:

```text
ΔHrx = -50,286 kJ/kmol
```

A second calculation uses:

```text
ΔHrx = -60,000 kJ/kmol
```

The latter is the value discussed in the manuscript from Garcia-Hernandez et al. (2019).

### Deterministic sensitivity

| Case | `ΔHrx` (kJ/kmol) | `Tmax` (K) | SAFE | Safe fraction |
|---|---:|---:|---:|---:|
| DWSIM / DIPPR | -50,286 | 391.83 | 594 / 729 | 81.5% |
| Garcia-Hernandez et al. (2019) | -60,000 | 403.77 | 549 / 729 | 75.3% |

The manuscript reports 45 of the 729 deterministic grid points changing classification between the two heat-of-reaction values.

### Monte Carlo sensitivity

| Case | `ΔHrx` (kJ/kmol) | `Tmax` (K) | SAFE | Safe fraction |
|---|---:|---:|---:|---:|
| DWSIM / DIPPR | -50,286 | 391.83 | 491 / 500 | 98.2% |
| Garcia-Hernandez et al. (2019) | -60,000 | 403.77 | 480 / 500 | 96.0% |

The same sampled operating points are reused for the two Monte Carlo enthalpy cases. The manuscript reports 11 of 500 Monte Carlo cases changing classification.

## Generated outputs

Running `code/Simcopy.py` produces the following outputs:

| File | Output |
|---|---|
| `stage2_v2_results.png` | Stage 2 PID simulation plots |
| `soe_sweep_results.csv` | Deterministic SOE result for each grid point |
| `soe_envelope_plot.png` | Deterministic SOE envelope plots |
| `monte_carlo_results.csv` | Monte Carlo case results |
| `probabilistic_safety_map.png` | Probabilistic safety map |
| `hrx_sensitivity_results.csv` | Deterministic heat-of-reaction comparison |
| `mc_hrx_sensitivity_results.csv` | Monte Carlo heat-of-reaction comparison |

The main result tables contain:

```text
T0
Ca0
Tc_supply
peak_T
settled_T
max_excursion_s
conversion
label
```

The heat-of-reaction sensitivity comparison contains:

```text
T0
Ca0
Tc_supply
label_dippr
label_lit
flipped
```

The repository does not contain separately exported raw result tables from the original work. The numerical outputs are generated by the supplied Python source, while the manuscript contains the reported aggregate results and figures.

# Output provenance

These seven files are companion artifacts prepared from the supplied manuscript and Python source.

The manuscript reports aggregate deterministic and Monte Carlo results and includes three deterministic SOE figures plus one probabilistic safety-map figure. It does **not** provide the complete 729-row deterministic CSV, the complete 500-row Monte Carlo CSV, or the point-by-point sensitivity CSVs as tabulated data. Those pointwise datasets therefore have not been invented here. The CSV files in this folder contain only values explicitly reported in the manuscript, with a `record_type`/`source` field identifying their status.

`stage2_v2_results.png` was regenerated from the supplied Python implementation because the Stage 2 plot itself is not embedded in the manuscript. The other three PNG figures are extracted from the manuscript artwork.

For a true pointwise `*_results.csv`, the original generated CSVs from the computational run are required, or the supplied Python model must be executed and its exact outputs retained.

## Reproducibility

The supplied Python script imports NumPy, Pandas, SciPy, and Matplotlib. No Python version is pinned by the source.

The main values to compare after a reproduction are:

| Check | Expected result |
|---|---:|
| Nominal steady-state temperature | 330.0 K |
| Baseline `Tmax` | 391.83 K |
| Baseline deterministic sweep | 594 / 729 SAFE |
| Deterministic sweep at `ΔHrx = -60,000 kJ/kmol` | 549 / 729 SAFE |
| Baseline Monte Carlo run | 491 / 500 SAFE |
| Monte Carlo at `ΔHrx = -60,000 kJ/kmol` | 480 / 500 SAFE |

Differences may occur because of dependency versions, numerical solver behavior, platform differences, or later edits to the code. This archive records the supplied computational snapshot and does not claim bit-for-bit reproducibility on every platform.

The DWSIM project is retained as supplied. Its embedded metadata identifies DWSIM build 9.0.5.0 and a save time of 2026-09-30 19:27:18.3915502 +05:30.

## Limitations

- The study is simulation-based and does not include laboratory or plant validation.
- It considers a single CSTR configuration.
- The uncertainty model is limited to selected kinetic parameters.
- Disturbance analysis is limited to feed temperature and coolant conditions.
- Effects such as fouling, sensor bias, actuator degradation, parameter drift, mechanical failure, pressure relief behavior, and human factors are outside the scope of the study.
- The deterministic and probabilistic safety fractions arise from different sampling schemes and should not be interpreted as directly comparable probabilities.
- The supplied source contains Windows-specific output paths that may need to be adjusted on other systems.

## Citation

Citation metadata is provided in `CITATION.cff`.

## License

CC BY 4.0. See `LICENSE`.

---

#### Presented By: Bihan Chakraborty