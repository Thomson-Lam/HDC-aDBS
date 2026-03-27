# 1) Gaps, benchmarking holes, and writing pitfalls

The biggest implementation gap is that the proposal says “simulate healthy/pathological, encode windows, compare controllers,” but the paper will live or die on how rigorously the simulator-to-dataset pipeline is defined. Right now, you still need to make explicit:

the simulator sampling rate seen by the controller,
how the variable-step ODE output becomes a fixed-rate LFP stream,
the window length and stride,
whether online filtering is causal,
how train/val/test splits are done at the trajectory level, not the window level,
and how thresholds are calibrated without leakage.

The most important empirical pitfall is window leakage. If you generate one long pathological trajectory, cut it into overlapping windows, and randomly split those windows, your offline HDC results will look better than they really are. Split by seed/run/trajectory first, then extract windows. Otherwise the train and test sets are nearly the same signal with slight overlap.

The second major gap is baseline fairness. Your HDC controller currently gets smoothing, hysteresis, and lockout. If the classical beta-threshold controller is just a bare “if beta > threshold then stimulate,” that is not a fair comparison. The clean comparison is:

same LFP stream,
same causal window length,
same decision cadence,
same stimulation epoch duration,
same lockout,
same on/off hysteresis structure if possible,

with the only difference being the decision statistic: beta power vs HDC margin. Your proposal already gives HDC hysteresis and epoch logic; mirror that structure for the classical baseline so you isolate the detector, not the controller mechanics.

A third weakness is that your final evaluation is still very beta-centric. You already separated offline labels from online operational labels, which is good, but in the write-up I would add at least one simulator-native secondary metric so the study is not judged only by the same biomarker the classical controller uses. Examples: synchrony index across STN cells, burst fraction, or distance to healthy PSD profile. That will strengthen the claim that HDC may capture richer pattern structure than a threshold on one band.

Another gap is stress testing specificity. You already included onset, recovery, and moderate-coupling windows, which is excellent. But you also need one explicit “healthy false-trigger” test: run long healthy simulations and report how often each adaptive controller stimulates unnecessarily. That matters a lot for the paper because “good suppression” is not enough if the controller chatters in healthy states.
