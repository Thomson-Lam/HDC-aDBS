# Overview 

Concrete steps that should break down the implementation plan:

1. Lock the experimental contract: controller-visible sampling rate, fixed-rate LFP stream, causal filtering, window/stride, stimulation epoch/lockout, and trajectory-level train/val/test splits before windowing
2. Build the ODE model module and data pipeline under that contract
3. Do a minimal open-loop stimulation sanity check early
4. Build the offline HDC pipeline
5. Run a small offline validator search for encoder settings
6. Freeze one encoder family and define matched beta-vs-HDC controller mechanics fairly
7. Calibrate beta and HDC thresholds on validation trajectories only
8. Offline robustness evaluation across regimes, including healthy false-trigger testing
9. Closed-loop controller evaluation
