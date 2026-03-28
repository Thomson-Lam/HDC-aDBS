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

Lock to 512 ms (128 samples at 250 Hz).

Reason:

* aligns with the global experiment spec
* reduces confounding between encoder design choices and environment contract
* keeps offline and online evaluation on the same temporal context

Normalization choices

Lock to per-window z-score only:

* subtract each window mean
* divide by each window standard deviation

Reason:

* least assumptions about global healthy reference statistics
* no dependency on reference-stat estimation quality
* naturally leakage-resistant in the offline pipeline

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
1 window size
1 normalization scheme
2 init methods

That is 18 configs before readout comparison.

For the locked MVP validator search:

D: 1000, 5000, 10000
bins: 8, 16
window: 512 ms
norm: per-window z-score
init: random, RFF

That is 12 configs, which is compact and easier to audit.

Selection metric for the validator

Use one score, not ten.

I would choose a validation ranking like:

primary: balanced accuracy or AUROC on healthy vs pathological
secondary: margin separation on onset / moderate windows
tie-breaker: mean encoding time per window
