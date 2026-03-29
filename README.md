# Project Proposal: HDC-aDBS Testbed

**A Lightweight Hyperdimensional Computing Controller for Adaptive Deep Brain Stimulation in an In Silico Parkinsonian Basal Ganglia Model**

---

## Implementation Quickstart

This repository currently includes a working HDC core stack for:

- bipolar hypervector primitives,
- random and RFF dictionary initialization,
- window encoder pipeline,
- prototype-similarity and linear-HDC readouts,
- fixed offline validator search,
- training layer with model save/load.

The current `train/valid-train.py` flow uses dummy data for end-to-end testing while the ODE-backed data pipeline is being integrated.

## High-level Architecture

The implemented path is:

1. choose/freeze encoder config from validator search,
2. encode windows into bipolar hypervectors,
3. train one readout variant:
   - prototype-similarity HDC, or
   - linear classifier over encoded HDC vectors,
4. evaluate and serialize artifacts.

## Repository Tree (current core)

```text
brain/
├── README.md
├── pyproject.toml
├── train/ # training code 
│   └── valid-train.py
├── hdc/ # hdc core 
│   ├── __init__.py
│   ├── initializers.py
│   ├── primitives.py
│   ├── dictionaries.py
│   ├── encoder.py # encoder classdef and utils 
│   ├── readouts.py
│   ├── training.py
│   └── search/
│       ├── __init__.py
│       ├── config.py
│       ├── validator.py
│       └── run.py 
├── tests/
│   ├── test_hdc_core.py # test HDC core utils 
│   ├── test_reproducibility_dummy.py 
│   ├── test_validator_search.py
│   └── test_training_layer.py
└── docs/
    ├── experiment-spec.md
    ├── encoder-searchspace.md
    ├── hdc.md
    └── models.md
```

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Run test suite:

```bash
uv run python -m unittest discover -s tests -v
```

## Run Validator + Training

Run the combined validator/training flow:

```bash
uv run python train/valid-train.py
```

What this script does:

1. looks for `artifacts/encoder_search/freeze_record.yaml`,
2. if missing, runs fixed validator search and creates freeze record,
3. trains both HDC readout variants with the frozen encoder config,
4. saves model artifacts and a compact training report.

Outputs:

- `artifacts/encoder_search/results.jsonl`
- `artifacts/encoder_search/leaderboard.csv`
- `artifacts/encoder_search/freeze_record.yaml`
- `artifacts/models/prototype/`
- `artifacts/models/linear/`
- `artifacts/models/train_report.yaml`

## How Prototype HDC Works

1. Raw window (length 128 samples at 250 Hz contract) is z-scored.
2. Samples are quantized into bins.
3. Value and position hypervectors are bound and bundled into one window HV.
4. Class prototypes are built by bundling train HVs per class.
5. Decision score is margin:

   `score = sim(h, P_path) - sim(h, P_healthy)`

6. Predict pathological if score >= threshold (default 0).

## How Linear HDC Works

1. Uses the same encoded window hypervectors as prototype HDC.
2. Fits a linear logistic model over HV features.
3. Uses linear decision function as pathological margin.
4. Predict pathological if score >= threshold (default 0).

## Notes on Current Stage

- The core HDC model stack is implemented and tested.
- The data pipeline for ODE trajectory sampling/splits is planned separately and can be connected later by replacing the dummy data provider in `train/valid-train.py`.

---

## 1. Overview

Deep Brain Stimulation (DBS) is an established therapy for Parkinson’s disease, but conventional systems typically operate in an open-loop fashion, delivering continuous stimulation regardless of whether pathological neural activity is present. While effective, this approach can waste energy and may contribute to unnecessary stimulation exposure.

This project proposes an **in silico adaptive DBS (aDBS) testbed** to evaluate whether a **lightweight Hyperdimensional Computing (HDC) encoder** can detect pathological beta-band synchrony in a simulated basal ganglia network and trigger stimulation only when needed. The goal is not to claim a clinical digital twin, but to build a controlled experimental platform for comparing an HDC-based controller against both **continuous DBS** and an **established classical aDBS baseline based on beta-band power thresholding**.

The central question is:

**Can an HDC-based state encoder achieve control efficacy comparable to a classical beta-threshold aDBS controller while maintaining a small computational and memory footprint?**

---

## 2. Motivation

Traditional continuous DBS is simple and effective, but it stimulates continuously even when pathological dynamics may not be present. Adaptive DBS addresses this by using neural biomarkers to determine when stimulation should be applied. In this project, we investigate whether HDC can serve as an alternative state representation for this decision process.

HDC is attractive because it offers:

* compact, high-dimensional representations
* robustness to noisy inputs
* simple similarity-based inference
* potentially lightweight runtime characteristics

Rather than positioning HDC as a replacement for existing clinical biomarkers, this project evaluates whether it can function as a practical and efficient control signal representation inside a simulated aDBS loop.

---

## 3. Project Objective

Develop an **in silico closed-loop Parkinsonian basal ganglia testbed** that:

1. simulates healthy and pathological network dynamics,
2. derives an LFP-like population observable,
3. uses HDC to classify present pathological network state,
4. triggers stimulation epochs when pathology is detected,
5. compares HDC-triggered aDBS against:

   * no stimulation,
   * continuous DBS,
   * and a classical beta-power threshold aDBS controller.

---

## 4. Research Question

The project is designed around the following primary research question:

**Can an HDC-based controller suppress pathological beta-state activity comparably to a classical beta-threshold aDBS controller, while using low stimulation and lightweight runtime resources?**

Secondary questions include:

* How separable are healthy and pathological simulated LFP windows under a simple HDC encoding scheme?
* How robust is the HDC controller to onset, recovery, and ambiguous transitional states?
* How much stimulation duty cycle reduction can adaptive control achieve relative to continuous DBS?
* What is the computational and memory footprint of the HDC controller compared with the classical baseline?

---

## 5. Methodology

### 5.1 Physiological Simulation

The simulation environment will be based on a coupled ODE model of the Parkinsonian basal ganglia network, centered on the **STN-GPe loop**, with the option to extend to STN-GPe-GPi-Thalamus if stable and tractable.

Two dynamical regimes will be generated:

* **Healthy regime:** normal STN-GPe coupling
* **Pathological regime:** elevated STN-GPe coupling producing pathological beta-band synchrony

A population-level **LFP surrogate** will be computed from the STN subsystem by summing membrane voltages or synaptic currents across the population.

The simulator will be implemented in Python using **NumPy** and **`scipy.integrate.solve_ivp`**, executed in a **chunked loop** to support online control updates and stimulation injection.

### 5.2 Ground Truth and Operational Labels

To avoid circular evaluation, offline and online labels are separated.

**Offline Ground Truth**
Used for training and separability analysis. Labels are defined strictly by the simulator’s parametric regime:

* healthy windows come from healthy-regime simulations
* pathological windows come from pathological-regime simulations

**Online Operational Label**
Used for real-time tracking and evaluation during closed-loop control. Pathology is operationally marked when the LFP surrogate’s **beta-band power (13–30 Hz)** exceeds a threshold defined as the **95th percentile of the healthy baseline**.

This separation ensures that HDC is not trained and evaluated solely on the same beta statistic used in the classical controller.

### 5.3 HDC State Encoder

The HDC classifier will use a simple, deliberately lightweight encoding pipeline:

1. normalize each LFP window
2. quantize amplitude values into bins
3. bind value hypervectors with position hypervectors
4. bundle across the window to form one window hypervector

Two class prototypes will be constructed:

* **Healthy prototype** ( \mathbf{P}_{healthy} )
* **Pathological prototype** ( \mathbf{P}_{path} )

For a given window hypervector ( \mathbf{h} ), the controller computes the confidence margin:

[
M_c = s(\mathbf{h}, \mathbf{P}*{path}) - s(\mathbf{h}, \mathbf{P}*{healthy})
]

where ( s(\cdot,\cdot) ) is a similarity measure such as cosine similarity.

A smoothed version of ( M_c ) will be used to stabilize decisions.

### 5.4 Transitional Robustness Testing

To ensure the HDC classifier does not merely memorize idealized extremes, it will be tested against three explicitly defined transitional regimes:

1. **Onset Windows**
   Windows surrounding the time where simulator parameters shift from healthy to pathological coupling.

2. **Recovery Windows**
   Windows occurring after stimulation, where the network transitions out of suppression and may return toward bursting or healthy behavior.

3. **Moderate-Coupling Windows**
   Static runs generated using intermediate synaptic couplings between the healthy and pathological extremes.

These conditions will stress-test separability and controller stability in ambiguous regions of state space.

### 5.5 Control Loop

The closed-loop controller will be implemented as a plain Python state machine.

**Decision Logic**

* stimulation turns on when smoothed ( M_c > \theta_{on} )
* stimulation can turn off only when ( M_c < \theta_{off} ), with ( \theta_{off} < \theta_{on} )

This introduces hysteresis and reduces flickering.

**Stimulation Logic**
When triggered, the controller delivers a **stimulation epoch**, not a single pulse. Each epoch consists of a **130 Hz pulse train** for a fixed duration, such as 200 ms, injected into the STN model.

A lockout period of approximately 50–100 ms will follow each epoch to prevent runaway retriggering and to allow the network response to be observed.

---

## 6. Baselines and Experimental Conditions

The core experiment compares four conditions using the same simulated pathological onset seed.

### Condition 1: No Stimulation

The network remains in the pathological regime without intervention.

### Condition 2: Continuous DBS

A 130 Hz stimulation train is delivered continuously throughout the run.

### Condition 3: Classical aDBS Baseline

An established classical controller triggers stimulation epochs based solely on **beta-band power threshold crossing**.

### Condition 4: HDC-Triggered aDBS

The proposed HDC margin-based controller triggers stimulation based on the smoothed confidence margin with hysteresis.

This four-way comparison is the key improvement that turns the project into a meaningful scientific evaluation rather than a simple control demo.

---

## 7. Evaluation Metrics

The project evaluates both **control efficacy** and **system efficiency**.

### 7.1 Control Efficacy

**Beta Power Suppression**
Mean beta-band power during the run.
Target: HDC should be comparable to the classical aDBS baseline.

**Pathological Occupancy**
Percentage of time spent in the operational pathological state.
Target: low occupancy, ideally below 10%.

**Suppression Latency**
Time from pathology onset to suppression.
Target: competitive with the classical baseline.

### 7.2 Stimulation Efficiency

**Energy Proxy**
Measured using total pulse count or stimulation duty cycle, assuming fixed pulse amplitude and width across active conditions.

[
E \propto \text{number of pulses delivered}
]

Target: substantially lower than continuous DBS.

### 7.3 System Efficiency

**Computational Overhead**
Mean decision time per window.

**Memory Footprint**
Approximate size of controller state and prototype storage.

Because a beta-power controller can also be extremely efficient, this project will not assume HDC is automatically faster. Instead, the aim is to measure whether HDC maintains a **small practical runtime footprint** while delivering competitive control quality.

---

## 8. Expected Contributions

This project is expected to contribute:

1. an **in silico aDBS testbed** for evaluating lightweight control logic,
2. a comparison between **HDC-based state encoding** and a **classical beta-threshold controller**,
3. an analysis of controller performance in **transitional neural states**,
4. a demonstration of whether HDC can provide a viable low-complexity control representation in a simulated Parkinsonian setting.

The project does not claim clinical validation or patient specificity. Its purpose is to establish a credible prototype and evaluation framework for future work.

---

## 9. Scope and Constraints

This is a **research prototype**, not a clinical system.

### In Scope

* simulated basal ganglia dynamics
* LFP surrogate generation
* offline HDC separability analysis
* closed-loop stimulation control
* comparison against classical and continuous baselines
* lightweight runtime analysis

### Out of Scope

* patient-specific tuning
* real neural recordings
* implant-grade hardware validation
* clinical claims about symptom outcomes
* literal tremor modeling unless directly encoded in the simulator

The project focuses on **pathological beta-state suppression**, not direct clinical tremor prediction or treatment claims.

---

## 10. Execution Plan

### Days 1–3: MVP Physics

* implement the simulator in chunked `solve_ivp` form
* establish healthy and pathological runs
* verify beta elevation in pathological LFP surrogate
* fallback to a 2-population STN-GPe model if needed

### Days 4–6: Offline HDC Classification

* build the simple HDC encoder
* generate healthy, pathological, and transitional windows
* construct prototypes
* evaluate margin separability
* validate correspondence between HDC margin and classical beta activity

### Days 7–8: Open-Loop Stimulation Validation

* inject continuous 130 Hz stimulation into the pathological model
* verify that stimulation suppresses pathological beta dynamics
* tune stimulation amplitude or injection target if needed

### Days 9–11: Closed-Loop Integration

* implement HDC hysteresis controller
* implement classical beta-threshold baseline controller
* generate timeline plots of LFP, beta power, and stimulation events
* check whether lockout is masking control failures

### Days 12–14: Final Comparison and Ablations

* run all four conditions on the same onset seed
* compute all metrics
* compare control efficacy, stimulation cost, and system efficiency
* run small ablations on smoothing, hysteresis, and lockout settings if time permits

---

## 11. Risks & Plans for this Proposal 

### Risk 1: Full model tuning becomes unstable, simulation tuning takes too long

**Mitigation:** fall back to an STN-GPe-only model that still produces reliable beta-like oscillations.

### Risk 2: Stimulation does not suppress pathological dynamics, the bad regime

**Mitigation:** tune injection amplitude, pulse width, epoch duration, or stimulation target before closing the loop.

### Risk 3: HDC fails to outperform or match the classical baseline, looks good offline but performs badly online 


- use transitional windows
- use smoothing + hysteresis 

**Mitigation:** treat this as a valid result. The study still yields a meaningful comparison and identifies where HDC helps or falls short.

### Risk 4: Runtime advantage is unclear, "Lighter weight" claim is not true for HDC 

**Mitigation:** report measured decision latency and memory footprint honestly, without assuming superiority.

This is the experimental component of the project; we can always swap out this model with a different one and tweak this depending on the results.

---

## 12. Summary

This project proposes a focused and technically coherent testbed for adaptive DBS control in a simulated Parkinsonian basal ganglia network. Its novelty does not lie in adaptive DBS itself, but in evaluating whether **Hyperdimensional Computing** can serve as a practical state representation for real-time pathological-state detection inside a closed-loop controller.

By comparing HDC-triggered stimulation against both **continuous DBS** and an **established classical beta-threshold aDBS baseline**, the project is positioned to answer a meaningful systems question:

**Can HDC deliver comparable control efficacy while maintaining low stimulation burden and a lightweight implementation footprint?**

Even if the answer is ultimately negative, the project remains valuable as a rigorous prototype and benchmarking framework.

---
