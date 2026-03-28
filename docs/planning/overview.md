# Overview 

Concrete steps that should break down the implementation plan:

1. Lock the experimental contract: controller-visible sampling rate, fixed-rate LFP stream, causal filtering, window/stride, stimulation epoch/lockout, and trajectory-level train/val/test splits before windowing
2. Build the ODE model module and data pipeline under that contract
3. Do a minimal open-loop stimulation sanity check early for the following criteria:
    - regime generation: ODE model produces at least 1 stable healthy run and 1 stable pathological run, pathological run shows stronger beta band oscillation in the controller-visible surrogate than the healthy run
    - signal pipeline correctness: exposed function returns a clean fixed-rate 250 Hz stream; windowing and timestamps line up with 50ms design choice and surrogate is stable (no NaNs, exploded values or discontinuities from chunk boundaries and resampling)
    - Stim effect exists: If you apply a fixed open-loop stimulation burst or continuous stimulation during a pathological run, the surrogate changes in the expected direction.Beta-band power should drop during or shortly after stimulation relative to the unstimulated pathological condition.
    - effect is reproducible: supression is not a one off fluke from a single seed; run a few seeds and trajectories and check whether the qualitative behavior is the same
    - Stim is not breaking the sim
    - Basic dose sanity:  At least one stimulation setting should do little, and at least one should noticeably suppress pathology.
    -  The beta metric computed from the surrogate tracks what you see in the time trace / PSD. If the trace looks calmer after stimulation, the beta metric should reflect that

"Pass if:
  - pathological > healthy in beta power
  - open-loop stimulation reduces pathological beta power
  - the effect is reproducible across a small number of runs"


4. Build the offline HDC pipeline
5. Run a small offline validator search for encoder settings
6. Freeze one encoder family and define matched beta-vs-HDC controller mechanics fairly
7. Calibrate beta and HDC thresholds on validation trajectories only
8. Offline robustness evaluation across regimes, including healthy false-trigger testing
9. Closed-loop controller evaluation
