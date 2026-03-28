# RFF Initialization Notes

## Concise summary

RFF initialization replaces arbitrary random basis hypervectors with binary hypervectors sampled from a correlated Gaussian process so their expected pairwise similarities match a chosen target matrix. In practice, we build a target similarity matrix `M` (for our case: quantized value bins), convert it to a Gaussian covariance with `sin((pi/2) * M)`, sample Gaussian features, and apply `sign` to obtain binary `{-1, +1}` hypervectors.

## Locked assumptions for this project

| Parameter | Assumption |
| --- | --- |
| Entity set for RFF init | Value/bin hypervectors only (quantized signal bins). Position hypervectors stay random binary. |
| Binary convention | Use bipolar hypervectors in `{-1, +1}`. |
| Bin centers | Normalize bin centers to `[0, 1]`. |
| Target similarity `M` | RBF kernel over bin centers: `M_ij = exp(- (c_i - c_j)^2 / (2 * sigma^2))`. |
| RBF bandwidth `sigma` | Fixed by bin spacing, not tuned: `sigma = 1 / (n_bins - 1)` for `n_bins > 1`. |
| Covariance construction | `Sigma_hat = sin((pi/2) * M)` elementwise. |
| PSD handling | Symmetrize and clip negative eigenvalues to `0` before sampling. |
| Sign tie rule | `sign(0)` maps to `+1`. |
| Output shape | One hypervector per entity: `(n_entities, D)`. |

Contents:

1. **What problem RFF init is trying to solve**
2. **How the construction works step by step**
3. **Why Gaussians show up**
4. **Why the sign operation gives binary hypervectors**
5. **Where the arcsin / sine math comes from**
6. **What "richer representation" means mathematically**
7. **What this means in practice for your project**

---

# 1. What problem is RFF init trying to solve?

The paper’s argument is:

> HDC quality depends on the similarity relationships its hypervectors can represent. 

Classical HDC usually initializes basis hypervectors independently at random. That means the pairwise similarities between basis vectors are mostly arbitrary. The paper shows this random-init procedure can restrict the similarity matrices that are achievable in expectation. 

So the goal becomes:

> instead of assigning basis hypervectors arbitrarily, construct them so their pairwise similarities match a desired similarity matrix (M). 

That is the whole point of the RFF-style initialization.

---

# 2. The target object: a similarity matrix

Suppose you have (n) entities you want to encode:

* feature bins
* pixel intensities
* signal quantization levels
* dictionary items

You define an (n \times n) matrix (M), where

[
M_{ij} = \text{desired similarity between entity } i \text{ and entity } j
]

So:

* (M_{ii} = 1) usually
* similar entities should have large positive similarity
* dissimilar entities should have smaller or negative similarity

The paper says: if we want HDC vectors to respect meaningful relationships, we should directly try to instantiate hypervectors whose pairwise similarities realize this matrix in expectation. 

---

# 3. The construction in the paper

Their algorithm is:

1. Start with target similarity matrix (M)
2. Compute
   [
   \hat{\Sigma} = \sin\left(\frac{\pi}{2} M\right)
   ]
   elementwise
3. Eigendecompose (\hat{\Sigma} = U \Lambda U^T)
4. Sample Gaussian vectors
5. Transform them using (U\Lambda^{1/2})
6. Apply `sign` elementwise
7. The result is a set of binary hypervectors 

The key theorem/lemma behind this is:

[
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)]
======================================================

\frac{2}{\pi}\arcsin(\mathbb{E}[XY])
]

for jointly Gaussian zero-mean unit-variance random variables (X,Y). 

This formula is the bridge between:

* **Gaussian correlation**
  and
* **expected binary similarity after sign-thresholding**

That is the heart of the method.

---

# 4. Let’s derive the logic slowly

## Step A: We want binary hypervectors

In binary HDC, each hypervector is something like

[
v_i \in {-1,+1}^D
]

and the similarity is usually normalized inner product:

[
S(v_i,v_j)=\frac{1}{D}\sum_{k=1}^D v_{ik}v_{jk}
]

So if coordinates agree often, similarity is high. 

The problem is: how do we generate these binary vectors so that

[
\mathbb{E}[S(v_i,v_j)] \approx M_{ij} , ?
]

---

## Step B: Binary variables are hard to control directly

If you try to directly generate correlated binary (\pm 1) variables with exactly the right correlation matrix, that is annoying and generally hard.

So the paper uses an easier route:

1. generate **correlated Gaussians**
2. threshold them with `sign`
3. get binary (\pm1) values

This is much easier because Gaussian correlation structure is well understood.

---

# 5. One coordinate at a time

Forget the full (D)-dimensional hypervector for a second.

Focus on a single coordinate (k). For that coordinate, we want to generate (n) binary values:

[
v_{1k}, v_{2k}, \dots, v_{nk}
\in {-1,+1}
]

one for each encoded entity.

The trick is:

* first sample Gaussian values
  [
  z_{1k}, z_{2k}, \dots, z_{nk}
  ]
  jointly from a multivariate Gaussian with a chosen covariance
* then set
  [
  v_{ik} = \operatorname{sgn}(z_{ik})
  ]

Do this independently for each coordinate (k=1,\dots,D).

Then each entity (i) gets a binary hypervector

[
v_i = (v_{i1},v_{i2},\dots,v_{iD})
]

---

# 6. Why choose Gaussian covariance as (\sin(\frac{\pi}{2}M))?

This is where the arcsin formula enters.

For two Gaussian variables (X,Y), the paper gives:

[
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)]
======================================================

\frac{2}{\pi}\arcsin(\mathbb{E}[XY])
]

If we want the expected binary similarity to equal (M_{ij}), then we want

[
M_{ij}
======

\frac{2}{\pi}\arcsin(\mathbb{E}[X_iX_j])
]

Solve for the Gaussian correlation:

[
\mathbb{E}[X_iX_j]
==================

\sin\left(\frac{\pi}{2}M_{ij}\right)
]

So if you define the Gaussian covariance matrix by

[
\hat{\Sigma}_{ij}
=================

\sin\left(\frac{\pi}{2}M_{ij}\right)
]

then after sign-thresholding, the expected binary inner products reproduce (M). 

That is the entire reason for the sine transform.

---

# 7. Why eigendecomposition appears

To sample a Gaussian vector with covariance (\hat{\Sigma}), you need a way to construct correlated samples.

If

[
\hat{\Sigma} = U\Lambda U^T
]

then a standard way to sample from (N(0,\hat{\Sigma})) is:

[
z = U\Lambda^{1/2}x
]

where

[
x \sim N(0,I)
]

That is why the algorithm does a symmetric eigendecomposition and uses (U\Lambda^{1/2}). 

It is just standard Gaussian sampling:

* start with independent standard normals
* linearly transform them
* obtain correlated normals with the covariance you want

---

# 8. What exactly the algorithm is outputting

Suppose there are (n) entities and you want hypervectors of dimension (d).

The algorithm samples (d) independent Gaussian draws in (\mathbb{R}^n). Each draw gives one coordinate across all (n) entities. Then sign-thresholding converts that coordinate into binary entries. Repeating this (d) times gives (n) binary hypervectors of length (d). 

So conceptually:

* each **column** = one hypervector dimension
* each **row** = one entity’s binary hypervector

As (d) gets large, empirical similarities converge to the target expected similarities.

---

# 9. A tiny example

Say you only want to encode 3 entities:

* A
* B
* C

and you want desired similarity matrix

[
M=
\begin{bmatrix}
1 & 0.8 & -0.2\
0.8 & 1 & 0.1\
-0.2 & 0.1 & 1
\end{bmatrix}
]

Then compute

[
\hat{\Sigma}*{ij} = \sin\left(\frac{\pi}{2}M*{ij}\right)
]

Roughly:

* if (M_{ij}=1), then (\sin(\pi/2)=1)
* if (M_{ij}=0.8), then covariance is (\sin(0.4\pi))
* if (M_{ij}=-0.2), then covariance is (\sin(-0.1\pi))

So now you have a Gaussian covariance matrix saying:

* A and B should be strongly positively correlated
* A and C slightly negatively correlated
* B and C mildly positively correlated

Then:

1. sample correlated Gaussian triples ((z_A,z_B,z_C))
2. take signs
3. repeat many times

After enough repetitions, the binary vectors for A, B, C will have dot products close to the desired (M).

So the binary vectors are no longer “random unrelated symbols.” They now encode a chosen geometry.

---

# 10. Why this is richer than naive random init

With naive random init:

* basis vectors are nearly orthogonal by default
* pairwise similarities mostly hover near zero
* the encoder does not respect closeness among feature values

With RFF-style init:

* you can deliberately make nearby or semantically related items more similar
* the basis dictionary reflects a kernel-like geometry
* the resulting hypervectors preserve more meaningful structure

So “richer” means:

> the family of achievable pairwise similarity relations is broader and better aligned with the data.  

That is a precise representational claim, not just a vague one.

---

# 11. Why they mention RBF kernels

The construction above still leaves one big question:

> what should the target matrix (M) be?

The paper says they use an RBF kernel in practice. 

For scalar feature values (x_i, x_j), a typical RBF kernel is

[
K(x_i,x_j)=\exp\left(-\frac{|x_i-x_j|^2}{2\sigma^2}\right)
]

This means:

* nearby values get similarity near 1
* far values get similarity near 0

So if you are encoding quantized bins, intensity levels, or signal amplitudes, an RBF-based (M) says:

> similar raw values should map to similar basis hypervectors

That is much more reasonable than treating every bin as unrelated.

---

# 12. Why this is called “RFF” at all

Classical random Fourier features approximate shift-invariant kernels by mapping inputs through random sinusoidal features. The paper is not using textbook RFF in the exact usual shallow-kernel-machine way; instead, it uses the same spirit:

* start from a desired kernel/similarity matrix
* generate a randomized feature representation that preserves that similarity structure

So the “RFF” label is really pointing to:

* kernel-inspired initialization
* preserving similarity through random feature machinery

Then they binarize it for HDC hardware efficiency.

---

# 13. Important condition: positive semidefinite issue

The paper notes that if

[
\sin\left(\frac{\pi}{2}M\right)
]

is positive semidefinite, then the algorithm can exactly achieve (M) in expectation; otherwise it gives an approximation. 

Why?

Because a covariance matrix must be positive semidefinite. If your transformed matrix is not PSD, it cannot be a valid covariance matrix for a Gaussian.

So exact realization depends on the transformed similarity matrix being a legal covariance matrix.

---

# 14. The full mathematical picture in one chain

Here is the whole pipeline mathematically:

### Desired:

Choose binary hypervectors (v_1,\dots,v_n \in {-1,1}^d) such that

[
\mathbb{E}[S(v_i,v_j)] \approx M_{ij}
]

### Trick:

For each dimension (k), sample

[
z^{(k)} \sim N(0,\hat{\Sigma})
]

with

[
\hat{\Sigma} = \sin\left(\frac{\pi}{2}M\right)
]

### Threshold:

Set

[
v_{ik} = \operatorname{sgn}(z_i^{(k)})
]

### Then:

By the Gaussian sign lemma,

[
\mathbb{E}[v_{ik}v_{jk}]
========================

# \frac{2}{\pi}\arcsin(\hat{\Sigma}_{ij})

\frac{2}{\pi}\arcsin\left(\sin\left(\frac{\pi}{2}M_{ij}\right)\right)
]

and on the relevant range this gives back (M_{ij}). 

Since full-vector similarity is the average across coordinates, the expected similarity converges to (M_{ij}).

---

# 15. Intuition for your DBS project

For your setting, think of the dictionary entries as things like:

* quantized beta power bins
* normalized signal values
* maybe other feature levels

With random init:

* neighboring bins are unrelated

With RFF init:

* neighboring bins can be made similar according to an RBF kernel

So if the real signal has continuity, smoothness, or gradual transitions, the hypervector codebook now reflects that geometry better.

That means your encoder is less likely to throw away useful structure before classification even starts.

---

# 16. The deepest intuition

The deepest idea is:

> binary HDC by itself is crude, but if you choose the binary vectors by thresholding a carefully correlated Gaussian process, you can “smuggle in” a much richer similarity geometry while still ending up with hardware-friendly binary hypervectors.

That is why this method is clever.

It keeps:

* binary vectors
* simple similarity
* HDC efficiency

but improves:

* representational geometry

---

# 17. One concise summary

RFF initialization works by choosing a target similarity matrix (M), converting it into a Gaussian covariance (\hat{\Sigma}=\sin(\frac{\pi}{2}M)), sampling correlated Gaussian features, and taking signs to produce binary hypervectors; the arcsin identity guarantees that the expected inner products of those binary vectors match the desired similarities, so the basis hypervectors encode a kernel-like geometry instead of arbitrary random unrelated symbols. 

Next, I can make this even more concrete by giving you a **small numerical worked example** with 3 entities and showing the sampled Gaussian vectors, sign step, and resulting binary similarities.
