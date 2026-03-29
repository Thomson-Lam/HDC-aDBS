# TODOs

## 1) Lock experiment rules (completed, pending review)

What this step does:
Freeze shared rules before coding model-specific logic so comparisons are fair and reproducible.

What is locked:

1. Fixed controller-visible signal contract:
   - sampling rate,
   - resampling boundary,
   - causal filtering policy,
   - window length and stride.
2. Fixed controller mechanics:
   - stimulation epoch duration,
   - post-stimulation lockout,
   - matched controller structure where possible.
3. Fixed data policy:
   - split by full trajectory/seed before windowing,
   - frozen ranking and freeze-before-test rules.

Status: COMPLETED. PENDING REVIEW.

## 2) Build simulation and signal pipeline (TODO)

What this step does:
Generate healthy and pathological trajectories and convert solver output into the exact fixed-rate stream seen by controllers.

Implementation units:

1. ODE model module for healthy and pathological regimes.
2. Chunked simulator runner with deterministic seeding.
3. LFP-like surrogate extraction from model state.
4. Resampling from variable-step solver output to fixed 250 Hz.
5. Signal safety checks:
   - no NaN/infinite values,
   - no timestamp reversals,
   - no discontinuity artifacts at chunk boundaries.

## 3) Run open-loop stimulation sanity gate (TODO)

What this step does:
Verify stimulation can suppress pathological behavior before building adaptive decision logic.

Checks to pass:

1. Regime generation:
   - healthy and pathological runs are stable,
   - pathological run has stronger beta-band activity than healthy.
2. Signal pipeline correctness:
   - stream is fixed-rate at 250 Hz,
   - windows align with 50 ms decision cadence,
   - no visible chunk-boundary artifacts.
3. Stimulation effect:
   - fixed bursts or continuous stimulation reduce pathological beta relative to unstimulated pathological baseline.
4. Reproducibility:
   - suppression behavior repeats across multiple seeds.
5. Dose sanity:
   - at least one weak stimulation setting has little effect,
   - at least one stronger setting gives noticeable suppression.
6. Metric consistency:
   - beta metric agrees with time-trace and power spectral density (PSD) behavior.

Pass criteria:

- pathological beta > healthy beta,
- open-loop stimulation reduces pathological beta,
- effect is reproducible across a small seed set.

## 4) Build HDC core library (Implemented, to review)

What this step does:
Create the reusable encoding/scoring components that turn one signal window into one hypervector and then into a decision statistic.

Implementation units:

1. Hypervector primitives:
   - binding function,
   - bundling function,
   - bipolar binarization/normalization helpers,
   - deterministic value and position hypervector dictionaries (using existing initializers).
2. Encoder pipeline:
   - per-window z-score normalization,
   - value quantization into bins,
   - position-value binding and bundle over a window,
   - output one hypervector per window.
3. Similarity and scoring:
   - cosine or normalized dot-product similarity,
   - margin score = similarity(pathological prototype) - similarity(healthy prototype).
4. Readouts:
   - prototype-similarity classifier,
   - linear classifier over encoded hypervectors,
   - shared prediction interface for both readout types.

## 4.5) Build HDC models that call the primitives (Done, to review)

Use one shared base + two concrete trainers:
- BaseHDCTrainer: owns encoder config, fit/predict interface, save/load hooks.
- PrototypeHDCTrainer: wraps WindowEncoder + PrototypeReadout.
- LinearHDCTrainer: wraps WindowEncoder + LinearReadout.

## 5) Data Pipeline: Build dataset and split layer (TODO AFTER ODE MODEL)

What this step does:
Provide clean, leak-free access to train/validation/test windows with regime labels.

Implementation units:

1. Data container for trajectory metadata, labels, and derived windows.
2. Splitter that operates by trajectory/seed (not by individual windows).
3. Window extraction for:
   - clean healthy,
   - clean pathological,
   - onset,
   - recovery,
   - moderate-coupling,
   - healthy-only specificity holdout.
4. Leakage guard checks to ensure overlapping windows never cross split boundaries.

## 6) Build offline validator and ranking engine (TODO)

What this step does:
Evaluate candidate encoders on hold-out validation data and rank them using one frozen rule to find the best encoder to freeze and use for both hypervector methods.

Implementation units:

1. Config generator for search options from `docs/hdc-encoder-optsearch.md`.
2. Per-config pipeline:
   - fit on train split,
   - evaluate on validation split only.
3. Frozen ranking rule:
   - primary metric: balanced accuracy on clean validation windows,
   - guardrail: minimum AUROC on onset and recovery validation windows,
   - tie-breaker: mean encoding time per window.
4. Structured outputs per config:
   - metrics,
   - guardrail pass/fail,
   - final rank,
   - runtime and memory stats.

## 7) Run selection funnel and freeze final HDC setup (TODO)

What this step does:
Narrow to a practical final design without touching test data.

Implementation units:

1. Keep top 2-3 validation-ranked candidates.
2. Run a small validation-only controller sweep for those candidates.
3. Freeze one final HDC setup:
   - encoder family and parameters,
   - readout choice,
   - thresholds,
   - smoothing,
   - hysteresis,
   - lockout.

Important rule:
No test-set performance may be used for ranking or freeze decisions.

## 8) Calibrate thresholds and run held-out offline report (TODO)

What this step does:
Finalize calibration on validation, then produce one untouched test report.

Implementation units:

1. Calibrate classical beta-threshold and HDC thresholds on validation trajectories only.
2. Run final offline report on test split once:
   - clean healthy vs pathological discrimination,
   - onset/recovery/moderate robustness,
   - healthy false-trigger behavior,
   - mean decision time per window,
   - memory footprint.

Important rule:
This stage is confirmatory reporting, not another optimization pass.

## 9) Integrate and evaluate closed-loop controllers (TODO)

What this step does:
Compare frozen HDC control against baselines under matched mechanics.

Implementation units:

1. Implement matched controller mechanics for classical and HDC paths.
2. Integrate frozen HDC decision statistic into controller state machine.
3. Run four-condition comparison:
   - no stimulation,
   - continuous DBS,
   - classical beta-threshold aDBS,
   - HDC-triggered aDBS.
4. Report locked metrics:
   - mean beta power,
   - pathological occupancy,
   - suppression latency,
   - stimulation duty cycle and pulse count,
   - healthy false-trigger rate,
   - mean decision time per window,
   - memory use.

## 10) Add reproducibility and test coverage (TODO)

What this step does:
Ensure results are trustworthy, repeatable, and auditable.

Implementation units:

1. Unit tests for HDC primitives, encoding shape/type invariants, and similarity math.
2. Unit tests for split leakage prevention and deterministic seed behavior.
3. Regression tests for ranking logic and guardrail enforcement.
4. Experiment-runner scripts plus saved config snapshots for exact reruns.
