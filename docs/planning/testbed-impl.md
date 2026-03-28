# What the project does, in plain English

**A simulated closed-loop DBS system that compares a standard beta-threshold controller against a lightweight HDC-based controller for detecting and suppressing Parkinson-like beta oscillations.**

1. **Brain simulator**

A mathematical model that produces neural signals. We will tune it so it can behave in two modes:

* a more normal mode
* a Parkinson-like abnormal oscillation mode

2. **State detector**

A lightweight algorithm that watches the simulated brain signal and decides whether the system is currently in a bad pathological state.

3. **Stimulator/controller**

If the detector thinks the brain has entered the bad state, it turns on simulated deep brain stimulation for a short burst to knock the system back out of that state.

Then we compare 4 strategies:

* do nothing
* always stimulate
* stimulate when classical beta-power threshold says to
* stimulate when our HDC detector says to

The real question is:

**Can our lightweight HDC detector control the system about as well as the standard beta-threshold method, while still being simple and cheap to run?**

---

# Layman analogy

"A smart thermostat for a machine that sometimes vibrates badly"

* The **simulator** is the machine.
* The **signal** is a sensor reading from the machine.
* The **detector** decides whether the machine is vibrating in a dangerous pattern.
* The **stimulator** is a corrective pulse that calms the machine down.

We want to know whether our custom detector can do as well as the standard detector.

---

# High-level implementation plan

## Phase 0: Lock the experimental contract

Before building anything else, define the exact signal path and evaluation protocol that every experiment will use.

### What we implement

* controller-visible sampling rate
* rule for converting variable-step ODE output into a fixed-rate LFP stream
* causal online filtering choices
* window length and stride
* decision cadence
* stimulation epoch duration and post-stim lockout
* train/validation/test split policy at the trajectory level, before window extraction
* threshold calibration policy using validation data only

### Output of this phase

* one reproducible protocol table used by both controllers
* one dataset policy that prevents leakage

### Why this matters

If the simulator-to-controller interface is underspecified, offline accuracy and online control results are hard to trust.

## Phase 1: Build the simulated brain

We first need a working mathematical simulation that can reliably produce:

* a normal regime
* an abnormal Parkinson-like regime with strong beta oscillations

### What we implement

* ODE model of the STN-GPe loop, with optional extension to more populations later
* chunked simulation loop so we can inject control decisions while the sim is running
* LFP-like population signal extracted from the model
* fixed-rate resampling pipeline that converts ODE output into the exact stream seen by controllers

### Output of this phase

* clean healthy signal
* clean pathological signal
* PSD plots showing beta-band is stronger in the pathological case
* validation that the healthy/pathological separation is preserved after the online signal pipeline

### Why this matters

If the simulation does not clearly produce the two regimes, everything downstream becomes meaningless.

---

## Phase 2: Build the HDC detector offline

Before doing any closed-loop control, we test whether the HDC encoder can even distinguish healthy vs pathological windows.

### What we implement

* signal windowing
* normalization
* quantization into bins
* HDC encoding:

  * value hypervectors
  * position hypervectors
  * binding
  * bundling
* healthy and pathological prototype vectors
* similarity scoring and margin:

  * similarity to healthy prototype
  * similarity to pathological prototype
  * margin = pathological similarity minus healthy similarity

### Data used

We generate multiple trajectories and split them by seed/run first, then extract windows within each split.

We test on:

* healthy windows
* pathological windows
* onset windows
* recovery windows
* moderate-coupling ambiguous windows

We also reserve long healthy-only trajectories for false-trigger evaluation later.

### Output of this phase

* histograms or plots showing class separation
* evidence that HDC margin correlates with beta activity
* held-out results that are not inflated by overlapping-window leakage

### Why this matters

This tells us whether the detector is learning something useful before we wire it into control.

---

## Phase 3: Validate stimulation in open loop

Before the controller decides when to stimulate, we must prove stimulation itself actually works.

### What we implement

* stimulation pulse train at 130 Hz
* fixed stimulation epoch, like 200 ms
* injection into the model as an external current/input

### Output of this phase

* traces showing pathological beta activity gets suppressed when stimulation is applied continuously

### Why this matters

If stimulation cannot suppress the bad state, there is no point building adaptive control logic.

---

## Phase 4: Build the closed-loop controllers

Now we connect the detectors to the simulator.

### Controller A: Classical baseline

* measure beta-band power from the LFP signal
* use the same causal window length and decision cadence as the HDC path
* smooth beta power over time
* use on/off hysteresis if possible
* apply the same stimulation epoch duration and lockout used by the HDC controller
* stimulate when beta power crosses calibrated thresholds

### Controller B: HDC controller

* compute HDC margin from the latest signal window
* smooth margin over time
* use hysteresis:

  * turn on above one threshold
  * turn off below a lower threshold
* apply lockout after each stimulation epoch

### Calibration rule

* calibrate beta thresholds and HDC thresholds on validation trajectories only
* freeze controller settings before any final test-set or closed-loop comparison

### Phase Output

* time plots showing:

  * simulated LFP
  * beta power
  * HDC margin
  * stimulation on/off timeline

Significance: This is the core system behavior.

---

## Phase 5: Run the 4-condition comparison

We run the same pathological scenario through all 4 strategies:

1. no stimulation
2. continuous DBS
3. classical beta-threshold aDBS
4. HDC-triggered aDBS

### Metrics we compute

#### Control performance

* average beta power
* pathological occupancy
* suppression latency

#### Specificity and robustness

* healthy false-trigger rate
* number of unnecessary stimulation epochs during healthy-only runs

#### Simulator-native secondary metrics

* synchrony index across model populations
* burst fraction
* distance to healthy PSD profile

#### Stimulation efficiency

* total pulses
* duty cycle

#### System efficiency

* decision time per window
* memory footprint

### Output of this phase

* final comparison table
* hero figure for presentation/report
* matched-controller comparison where the decision statistic is the main difference between methods

---

| Name                                                        | Usage Purpose                                                                                                                                                                                                                                                                    |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Python**                                                  | Primary language for the entire project. Used for simulation, control logic, HDC implementation, signal processing, experiments, and visualization.                                                                                                                              |
| **NumPy**                                                   | Core numerical computation library. Used for array operations, vectorized math, HDC hypervector operations (binding/bundling), signal windowing, and metric calculations.                                                                                                        |
| **SciPy (`integrate.solve_ivp`)**                           | ODE solver used to implement the **Terman-Rubin / STN-GPe neural dynamics simulator** that produces healthy and pathological regimes.                                                                                                                                            |
| **SciPy (`signal`)**                                        | Classical signal-processing utilities used to compute **power spectral density (PSD)**, bandpass filtering, and **beta-band power (13–30 Hz)** for the classical aDBS baseline controller.                                                                                       |
| **Hyperdimensional Computing (HDC)**                        | Lightweight representation learning method used for the **pathological state classifier**. Encodes LFP signal windows into high-dimensional vectors and compares them to healthy/pathological prototypes using similarity measures. Implemented manually with NumPy for the MVP. |
| **Terman-Rubin Basal Ganglia Model**                        | Biophysical ODE model of basal ganglia dynamics used to simulate neural activity in healthy vs Parkinsonian regimes. Provides the simulated neural signals that the controllers monitor and act upon.                                                                            |
| **STN-GPe Network Simulator**                               | Simplified neural circuit implementation derived from the Terman-Rubin framework. Generates beta-band oscillatory dynamics characteristic of Parkinsonian states. Serves as the environment for testing adaptive DBS controllers.                                                |
| **Classical Beta-Threshold Controller**                     | Baseline adaptive DBS algorithm using beta-band power from the LFP surrogate. Triggers stimulation epochs when beta power exceeds a threshold derived from healthy baseline statistics.                                                                                          |
| **HDC Margin-Based Controller**                             | Proposed adaptive controller using similarity margin between healthy and pathological hypervector prototypes. Uses smoothing and hysteresis to determine stimulation triggers.                                                                                                   |
| **Matplotlib**                                              | Primary visualization tool for plots such as LFP traces, power spectral density graphs, controller timelines, and comparison figures.                                                                                                                                            |
| **Jupyter Notebook**                                        | Interactive development and analysis environment used for experiment exploration, debugging, and producing final evaluation figures and reports.                                                                                                                                 |
| **Pandas (optional)**                                       | Lightweight tabular data handling for aggregating experiment metrics and generating comparison tables. Not required but useful for summarizing results.                                                                                                                          |
| **Python Scripts (`experiments/`)**                         | Scripts used to run simulation experiments across the four evaluation conditions: no stimulation, continuous DBS, classical aDBS, and HDC-triggered aDBS.                                                                                                                        |
| **Configuration Files (Python or YAML)**                    | Store simulation parameters, stimulation parameters, and HDC configuration settings to allow reproducible experiment runs.                                                                                                                                                       |
| **Synthetic Simulation Data**                               | Generated directly from the neural simulator. Used for training HDC prototypes and evaluating controllers. No external datasets are required.                                                                                                                                    |
| **Time Benchmarking (`time`, `timeit`)**                    | Used to measure **decision latency per window** for system efficiency evaluation.                                                                                                                                                                                                |
| **Memory Inspection (`sys.getsizeof`, `pympler` optional)** | Used to estimate memory footprint of HDC prototypes and controller state for system-efficiency analysis.                                                                                                                                                                         |
| **Version Control (Git)**                                   | Tracks development of simulator, controllers, and experiment scripts. Enables reproducibility and collaboration among engineers.                                                                                                                                                 |
| **Database (Not Required for MVP)**                         | No database required during development. Simulation outputs can be stored as NumPy arrays or CSV files. A lightweight database (e.g., SQLite) could optionally be added later for experiment tracking if the project evolves into a larger research platform.                    |
| **Deployment Layer (Optional Future)**                      | If later deployed as a demonstration platform, a lightweight API (FastAPI) and dashboard (Streamlit) could expose the simulator and controllers for interactive testing. Not required for the MVP.                                                                               |
---

# Suggested repo structure

```text
hdc-adbs-testbed/
├── README.md
├── requirements.txt
├── configs/
│   ├── sim_config.py
│   ├── stim_config.py
│   └── hdc_config.py
├── src/
│   ├── simulation/
│   │   ├── model.py
│   │   ├── dynamics.py
│   │   ├── lfp.py
│   │   └── runner.py
│   ├── stimulation/
│   │   ├── pulse_train.py
│   │   └── injector.py
│   ├── hdc/
│   │   ├── hypervectors.py
│   │   ├── encoder.py
│   │   ├── prototypes.py
│   │   └── classifier.py
│   ├── controllers/
│   │   ├── classical_beta_controller.py
│   │   ├── hdc_controller.py
│   │   └── state_machine.py
│   ├── analysis/
│   │   ├── beta_power.py
│   │   ├── metrics.py
│   │   └── plots.py
│   └── utils/
│       ├── windowing.py
│       └── timing.py
├── notebooks/
│   ├── 01_sim_validation.ipynb
│   ├── 02_hdc_offline_eval.ipynb
│   ├── 03_open_loop_stim.ipynb
│   └── 04_closed_loop_comparison.ipynb
└── experiments/
    ├── run_baselines.py
    └── run_ablation.py
```
---

# Success Criteria 

A successful project does not need to prove HDC is better in every way.

Success is:

* the simulator reliably enters a pathological beta state
* stimulation can suppress it
* the classical controller works
* the HDC controller also works
* you compare them fairly
* you can clearly say whether HDC is competitive, worse, or better

That is already a strong engineering result.

---

# Main technical risks

## Risk 1: simulation tuning takes too long

Fix:

* simplify to STN-GPe only
* stop chasing biological perfection

## Risk 2: stimulation does not suppress the bad regime

Fix:

* tune amplitude, pulse width, target population, epoch duration

## Risk 3: HDC classifier looks good offline but fails online

Fix:

* use transitional windows
* split train/validation/test by trajectory before windowing
* use smoothing + hysteresis
* inspect recovery behavior carefully

## Risk 4: comparison is unfair between controllers

Fix:

* match windowing, smoothing, hysteresis, epoch duration, and lockout across controllers
* calibrate thresholds on validation data only

## Risk 5: “lighter weight” claim is not true

Fix:

* do not assume it
* measure it honestly

---

# What each controller is really doing

## Classical beta-threshold controller

This is the standard rule-based approach.

It says:

* “If beta activity is above threshold, stimulate.”

Very interpretable and simple.

## HDC controller

This is a pattern-based approach.

It says:

* “This recent signal pattern looks more like the pathological prototype than the healthy prototype, so stimulate.”

It may capture richer temporal structure than a simple beta threshold.

---

# MVP goals 

* 2-population STN-GPe simulator
* one LFP surrogate
* one classical beta-threshold controller
* one simple HDC controller
* one comparison notebook
* one final table with metrics
