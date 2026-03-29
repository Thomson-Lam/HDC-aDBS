"""STN-GPe ODE right-hand side (Terman & Rubin 2002).

This file defines stn_gpe_rhs(), the function passed to scipy's solve_ivp.
It computes the time derivative dy/dt for the full network state vector y.

State vector layout
-------------------
STN cells come first, GPe cells follow.  Within each population, cells are
laid out consecutively, each occupying STN_STATE_DIM (or GPE_STATE_DIM) slots:

    y = [ V_s0, h_s0, n_s0, r_s0, Ca_s0, sA_0,   ← STN cell 0
          V_s1, h_s1, n_s1, r_s1, Ca_s1, sA_1,   ← STN cell 1 (if n_STN > 1)
          ...
          V_g0, h_g0, n_g0, r_g0, Ca_g0, sG_0,   ← GPe cell 0
          V_g1, h_g1, n_g1, r_g1, Ca_g1, sG_1,   ← GPe cell 1 (if n_GPe > 1)
          ... ]

Variable glossary
-----------------
V      : membrane voltage (mV)
h      : sodium inactivation gate  — tracks how many Na channels are available
n      : potassium activation gate — tracks how many K channels are open
r      : T-type Ca inactivation gate — tracks low-threshold Ca channel availability
Ca     : intracellular calcium concentration (mM)
s_AMPA : STN synaptic release gate  — how much glutamate is currently released
s_GABA : GPe synaptic release gate  — how much GABA is currently released

Gating variable dynamics
------------------------
All gating variables follow the Hodgkin-Huxley form:

    dx/dt = phi * (x_inf(V) - x) / tau_x(V)

where:
  x_inf(V) — steady-state value at voltage V  (Boltzmann sigmoid)
  tau_x(V) — voltage-dependent time constant  (larger = slower)
  phi      — overall speed scaling constant

x_inf(V) = 1 / (1 + exp(-(V - theta) / sigma))
  theta: voltage of 50% activation (mV)
  sigma: positive → activates with depolarisation
         negative → inactivates with depolarisation  ← used for h, r
"""

import numpy as np

from configs.sim_config import SimConfig, STN_STATE_DIM, GPE_STATE_DIM


# ---------------------------------------------------------------------------
# Boltzmann sigmoid helper
# ---------------------------------------------------------------------------

def _sigm(V, theta: float, sigma: float):
    """
    Boltzmann sigmoid: x_inf(V) = 1 / (1 + exp(-(V - theta) / sigma))

    Parameters
    ----------
    V     : float or ndarray — membrane voltage (mV)
    theta : float            — half-activation voltage (mV)
    sigma : float            — slope; positive = activates on depol,
                               negative = inactivates on depol

    The exponent is clipped to [-500, 500] before calling exp() to prevent
    overflow RuntimeWarnings.  At |x| > 500 the sigmoid is already saturated
    at 0 or 1 to full float64 precision, so clipping has no effect on the
    computed value — it only suppresses the warning.
    """
    x = -(V - theta) / sigma
    return 1.0 / (1.0 + np.exp(np.clip(x, -500.0, 500.0)))


# ---------------------------------------------------------------------------
# STN gating functions  (Terman & Rubin 2002, Table I)
# ---------------------------------------------------------------------------
# Naming: _stn_X_inf  → steady-state of gate X at voltage V
#         _stn_tau_X  → time constant of gate X at voltage V

# m — sodium activation (so fast it never lags; treated as instantaneous)
def _stn_m_inf(V):  return _sigm(V, -30.0,  15.0)

# h — sodium inactivation  (sigma < 0 → closes when depolarised)
def _stn_h_inf(V):  return _sigm(V, -39.0,  -3.1)
def _stn_tau_h(V):
    # Large tau at hyperpolarised Vs → h recovers slowly from inactivation.
    # tau_h ranges from ~1 ms (depolarised) to ~500 ms (very hyperpolarised).
    return 1.0 + 500.0 / (1.0 + np.exp(np.clip((V + 57.0) / 3.0, -500.0, 500.0)))

# n — potassium activation  (sigma > 0 → opens when depolarised)
def _stn_n_inf(V):  return _sigm(V, -32.0,   8.0)
def _stn_tau_n(V):
    # Peaks around V = -80 mV; gives delayed rectifier its characteristic delay.
    return 1.0 + 100.0 / (1.0 + np.exp((V + 80.0) / 26.0))

# r — T-type Ca inactivation  (sigma < 0 → more inactivated at depol voltages)
# r ≈ 1 at rest/hyperpol (channel available); r ≈ 0 at depol (channel blocked)
def _stn_r_inf(V):  return _sigm(V, -67.0,  -2.0)
def _stn_tau_r(V):
    # Slow recovery (~7-25 ms): cell must stay hyperpolarised long enough
    # to de-inactivate r before a T-type Ca burst can be triggered.
    return 7.1 + 17.5 / (1.0 + np.exp((V + 68.0) / 2.2))

# a — T-type Ca activation  (instantaneous; activates at low voltages ~-63 mV)
def _stn_a_inf(V):  return _sigm(V, -63.0,   7.8)

# s — high-threshold Ca activation  (instantaneous; activates at higher voltages)
def _stn_s_inf(V):  return _sigm(V, -39.0,   8.0)


# ---------------------------------------------------------------------------
# GPe gating functions  (Terman & Rubin 2002, Table II)
# ---------------------------------------------------------------------------

# m — sodium activation (instantaneous)
def _gpe_m_inf(V):  return _sigm(V, -37.0,  10.0)

# h — sodium inactivation
def _gpe_h_inf(V):  return _sigm(V, -58.0, -12.0)
def _gpe_tau_h(V):
    # GPe h recovers faster than STN (different theta/sigma).
    return 1.0 + 500.0 / (1.0 + np.exp((V + 40.0) / 12.0))

# n — potassium activation
def _gpe_n_inf(V):  return _sigm(V, -50.0,  14.0)
def _gpe_tau_n(V):
    return 1.0 + 100.0 / (1.0 + np.exp((V + 40.0) / 14.0))

# r — T-type Ca inactivation
def _gpe_r_inf(V):  return _sigm(V, -70.0,  -2.0)
def _gpe_tau_r(V):
    # Slower baseline (40 ms) than STN; GPe bursts need longer hyperpolarisation.
    return 40.0 + 17.5 / (1.0 + np.exp(np.clip((V + 68.0) / 2.2, -500.0, 500.0)))

# a — T-type Ca activation (instantaneous)
def _gpe_a_inf(V):  return _sigm(V, -57.0,   2.0)

# s — high-threshold Ca activation (instantaneous)
def _gpe_s_inf(V):  return _sigm(V, -35.0,   2.0)


# ---------------------------------------------------------------------------
# Synaptic release gate helper
# ---------------------------------------------------------------------------

def _syn_inf(V, theta: float, sigma: float):
    """
    Smooth Heaviside approximation for synaptic release probability.

    Returns a value near 0 when the presynaptic cell is at rest and
    near 1 when it is depolarised past the release threshold theta.
    The sigmoid is intentionally sharp (small sigma) to mimic the
    near-digital nature of vesicle release at the peak of an action potential.
    """
    return _sigm(V, theta, sigma)


# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------

def stn_gpe_rhs(t: float, y: np.ndarray, config: SimConfig, stim_fn=None) -> np.ndarray:
    """
    Compute dy/dt for the full STN-GPe network at time t.

    This function is passed directly to scipy.integrate.solve_ivp as the
    'fun' argument.  It must be fast — the solver calls it thousands of
    times per simulated second.

    Parameters
    ----------
    t        : float     — current simulation time (ms); needed for stim_fn
    y        : ndarray   — current state vector, shape (n_STN*6 + n_GPe*6,)
    config   : SimConfig — all biophysical and coupling parameters
    stim_fn  : callable(t) -> float, optional
               External stimulation current added to all STN cells (μA/cm²).
               Pass None for zero stimulation (default).  This is the
               hook used by open-loop (Phase 3) and closed-loop (Phase 4)
               controllers.

    Returns
    -------
    dydt : ndarray, same shape as y — time derivatives
    """
    dydt = np.empty_like(y)

    sp  = config.stn   # STN parameters
    gp  = config.gpe   # GPe parameters
    syn = config.syn   # synaptic parameters

    # External stimulation current this timestep (0 if no controller attached)
    I_stim = stim_fn(t) if stim_fn is not None else 0.0

    # ------------------------------------------------------------------
    # Index boundaries in the flat state vector
    # ------------------------------------------------------------------
    # All STN state vars occupy y[0 : gpe_offset]
    # All GPe state vars occupy y[gpe_offset : total]
    gpe_offset = config.n_STN * STN_STATE_DIM
    total      = gpe_offset + config.n_GPe * GPE_STATE_DIM

    # ------------------------------------------------------------------
    # Unpack STN state variables
    # ------------------------------------------------------------------
    # Slicing with step=STN_STATE_DIM extracts one variable across all cells.
    # e.g. for n_STN=2: y[0:12:6] = [V_s0, V_s1]
    #
    # Each of these is a 1-D array of length n_STN.
    V_s  = y[stn_var(0, config.n_STN, gpe_offset)]   # membrane voltage
    h_s  = y[stn_var(1, config.n_STN, gpe_offset)]   # Na inactivation
    n_s  = y[stn_var(2, config.n_STN, gpe_offset)]   # K activation
    r_s  = y[stn_var(3, config.n_STN, gpe_offset)]   # T-Ca inactivation
    Ca_s = y[stn_var(4, config.n_STN, gpe_offset)]   # intracellular Ca
    sA   = y[stn_var(5, config.n_STN, gpe_offset)]   # AMPA release gate (STN output)

    # ------------------------------------------------------------------
    # Unpack GPe state variables
    # ------------------------------------------------------------------
    V_g  = y[gpe_var(0, config.n_GPe, gpe_offset, total)]
    h_g  = y[gpe_var(1, config.n_GPe, gpe_offset, total)]
    n_g  = y[gpe_var(2, config.n_GPe, gpe_offset, total)]
    r_g  = y[gpe_var(3, config.n_GPe, gpe_offset, total)]
    Ca_g = y[gpe_var(4, config.n_GPe, gpe_offset, total)]
    sG   = y[gpe_var(5, config.n_GPe, gpe_offset, total)]   # GABA release gate (GPe output)

    # ------------------------------------------------------------------
    # STN ionic currents
    # ------------------------------------------------------------------
    # Instantaneous gates (no ODE needed — evaluated directly from V)
    m_s = _stn_m_inf(V_s)   # Na activation
    a_s = _stn_a_inf(V_s)   # T-Ca activation
    ss  = _stn_s_inf(V_s)   # high-thresh Ca activation

    I_L_s   = sp.g_L  * (V_s - sp.E_L)                        # passive leak
    I_Na_s  = sp.g_Na * m_s**3 * h_s * (V_s - sp.E_Na)        # fast Na spike
    I_K_s   = sp.g_K  * n_s**4 * (V_s - sp.E_K)               # K repolarisation
    I_T_s   = sp.g_T  * a_s**3 * r_s * (V_s - sp.E_Ca)        # T-Ca burst
    I_Ca_s  = sp.g_Ca * ss**2  * (V_s - sp.E_Ca)              # high-thresh Ca
    I_AHP_s = sp.g_AHP * (Ca_s / (Ca_s + sp.k1)) * (V_s - sp.E_K)  # Ca-dep K

    # ------------------------------------------------------------------
    # GPe→STN inhibitory synaptic current
    # ------------------------------------------------------------------
    # sG is the GPe release gate.  We average over all GPe cells so that
    # the total input to each STN cell is the same regardless of n_GPe.
    # config.g_GABA is the regime-dependent conductance weight.
    I_GABA_s = config.g_GABA * sG.mean() * (V_s - syn.E_GABA)

    # ------------------------------------------------------------------
    # STN derivatives
    # ------------------------------------------------------------------
    # dV/dt = (sum of all currents) / C
    # Currents that push V up appear as negative in the convention
    # I_ion = g * (V - E_rev) because when V < E_rev the current is inward.
    dV_s = (
        - I_L_s - I_Na_s - I_K_s - I_T_s - I_Ca_s - I_AHP_s
        - I_GABA_s                   # inhibitory input from GPe
        + sp.I_app                   # tonic excitatory drive
        + I_stim                     # optional controller stimulation
    ) / sp.C

    # Gating variable ODEs: phi * (x_inf - x) / tau
    # phi values: h=0.75, n=0.75, r=0.5  (from Terman & Rubin 2002)
    dh_s  = 0.75 * (_stn_h_inf(V_s) - h_s) / _stn_tau_h(V_s)
    dn_s  = 0.75 * (_stn_n_inf(V_s) - n_s) / _stn_tau_n(V_s)
    dr_s  = 0.50 * (_stn_r_inf(V_s) - r_s) / _stn_tau_r(V_s)

    # Calcium: rises with Ca currents, decays exponentially via extrusion
    dCa_s = sp.eps * (-I_Ca_s - I_T_s - sp.k_Ca * Ca_s)

    # STN AMPA gate dynamics: opens when STN fires, drives GPe
    H_s  = _syn_inf(V_s, syn.theta_syn_s, syn.sigma_syn_s)  # release probability
    dsA  = syn.alpha_AMPA * (1.0 - sA) * H_s - syn.beta_AMPA * sA

    # ------------------------------------------------------------------
    # GPe ionic currents
    # ------------------------------------------------------------------
    m_g = _gpe_m_inf(V_g)
    a_g = _gpe_a_inf(V_g)
    sg  = _gpe_s_inf(V_g)

    I_L_g   = gp.g_L  * (V_g - gp.E_L)
    I_Na_g  = gp.g_Na * m_g**3 * h_g * (V_g - gp.E_Na)
    I_K_g   = gp.g_K  * n_g**4 * (V_g - gp.E_K)
    I_T_g   = gp.g_T  * a_g**3 * r_g * (V_g - gp.E_Ca)
    I_Ca_g  = gp.g_Ca * sg**2  * (V_g - gp.E_Ca)
    I_AHP_g = gp.g_AHP * (Ca_g / (Ca_g + gp.k1)) * (V_g - gp.E_K)

    # ------------------------------------------------------------------
    # STN→GPe excitatory synaptic current
    # ------------------------------------------------------------------
    # sA is the STN release gate; averaged over all STN cells.
    I_AMPA_g = syn.g_AMPA * sA.mean() * (V_g - syn.E_AMPA)

    # ------------------------------------------------------------------
    # GPe derivatives
    # ------------------------------------------------------------------
    dV_g = (
        - I_L_g - I_Na_g - I_K_g - I_T_g - I_Ca_g - I_AHP_g
        - I_AMPA_g                   # excitatory input from STN
        + gp.I_app                   # tonic drive
    ) / gp.C

    # phi values for GPe: h=0.05, n=0.05, r=0.5  (Terman & Rubin 2002)
    dh_g  = 0.05 * (_gpe_h_inf(V_g) - h_g) / _gpe_tau_h(V_g)
    dn_g  = 0.05 * (_gpe_n_inf(V_g) - n_g) / _gpe_tau_n(V_g)
    dr_g  = 0.50 * (_gpe_r_inf(V_g) - r_g) / _gpe_tau_r(V_g)

    dCa_g = gp.eps * (-I_Ca_g - I_T_g - gp.k_Ca * Ca_g)

    # GPe GABA gate dynamics: opens when GPe fires, drives STN
    H_g  = _syn_inf(V_g, syn.theta_syn_g, syn.sigma_syn_g)
    dsG  = syn.alpha_GABA * (1.0 - sG) * H_g - syn.beta_GABA * sG

    # ------------------------------------------------------------------
    # Pack derivatives back into dydt in the same interleaved layout as y
    # ------------------------------------------------------------------
    dydt[stn_var(0, config.n_STN, gpe_offset)] = dV_s
    dydt[stn_var(1, config.n_STN, gpe_offset)] = dh_s
    dydt[stn_var(2, config.n_STN, gpe_offset)] = dn_s
    dydt[stn_var(3, config.n_STN, gpe_offset)] = dr_s
    dydt[stn_var(4, config.n_STN, gpe_offset)] = dCa_s
    dydt[stn_var(5, config.n_STN, gpe_offset)] = dsA

    dydt[gpe_var(0, config.n_GPe, gpe_offset, total)] = dV_g
    dydt[gpe_var(1, config.n_GPe, gpe_offset, total)] = dh_g
    dydt[gpe_var(2, config.n_GPe, gpe_offset, total)] = dn_g
    dydt[gpe_var(3, config.n_GPe, gpe_offset, total)] = dr_g
    dydt[gpe_var(4, config.n_GPe, gpe_offset, total)] = dCa_g
    dydt[gpe_var(5, config.n_GPe, gpe_offset, total)] = dsG

    return dydt


# ---------------------------------------------------------------------------
# Index helpers  (keep the slicing logic readable in stn_gpe_rhs)
# ---------------------------------------------------------------------------

def stn_var(var_idx: int, n_STN: int, gpe_offset: int) -> slice:
    """
    Return a slice that selects variable var_idx across all STN cells.

    Example: stn_var(0, n_STN=2, gpe_offset=12) → slice(0, 12, 6)
    which picks y[0] and y[6] — the voltage of each STN cell.
    """
    start = var_idx                       # first cell's variable
    stop  = gpe_offset                    # STN block ends here
    step  = STN_STATE_DIM                 # jump to same variable in next cell
    return slice(start, stop, step)


def gpe_var(var_idx: int, n_GPe: int, gpe_offset: int, total: int) -> slice:
    """
    Return a slice that selects variable var_idx across all GPe cells.
    """
    start = gpe_offset + var_idx
    stop  = total
    step  = GPE_STATE_DIM
    return slice(start, stop, step)
