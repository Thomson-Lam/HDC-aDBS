# A concrete small validator search

Requirements:
- small enough to finish in time
- meaningful enough to choose a sane encoder
- frozen before the final closed-loop experiments


## 1. First fix two infrastructure choices

Before tuning, keep fixed:

- resampled LFP rate: 250 Hz 
- decision stride: 50 ms

## 2. Search dimensions


Dimension: D = {1000, 5000, 10000}

Bins: bins = {8, 16, 32}

Window length:

Because your target is beta-band pathology, the window has to be long enough to reflect oscillatory content, not just pointwise amplitude.

At 250 Hz, I would search:

256 ms
512 ms
768 ms

That is about:

64 samples
128 samples
192 samples

Why this range:

too short and the window may not capture enough beta structure
too long and the detector becomes sluggish

If you want an even smaller search, do just:

256 ms
512 ms
Normalization choices

Use exactly these 3 options:

per-window z-score
subtract window mean, divide by window std
healthy-reference z-score
subtract healthy-train mean, divide by healthy-train std
robust scaling + clipping
center by median, scale by IQR or MAD, then clip to a fixed range before binning

My recommendation:

start with per-window z-score
keep healthy-reference z-score as the most important alternative
robust scaling is optional if the simulated amplitudes vary a lot
Initialization choice

Use:

random binary
RFF-based correlated initialization

That is the paper-inspired ablation worth doing. The paper explicitly argues that classical random initialization restricts the similarity structures binary HDC can realize, and proposes a more principled RFF construction instead.

Readout choice

For each frozen encoder config, compare:

prototype similarity
linear classifier over encoded hypervectors

This is important because the paper also shows that better encoding and better readout are separate issues. If the class is heterogeneous, a single bundled prototype can smear structure.

What to keep fixed during validator search

Do not search everything.

Keep fixed:

same position encoding scheme
same value binding rule
same similarity metric
same train/val/test split
same stride

For the MVP:

binary hypervectors
XOR / sign-product style binding
cosine or normalized dot-product similarity
Suggested search budget

A practical budget is:

3 dimensions
3 bin counts
2 window sizes
2 normalization schemes
2 init methods

That is 72 configs before readout comparison.

That is already borderline, so I would shrink it to:

D: 1000, 5000, 10000
bins: 8, 16
windows: 256 ms, 512 ms
norm: per-window z, healthy-reference z
init: random, RFF

That is 48 configs, which is much more realistic.

Selection metric for the validator

Use one score, not ten.

I would choose a validation ranking like:

primary: balanced accuracy or AUROC on healthy vs pathological
secondary: margin separation on onset / moderate windows
tie-breaker: mean encoding time per window

