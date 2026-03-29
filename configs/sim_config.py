"""Simulation configuration for the STN-GPe ODE model.

The STN (subthalamic nucleus) and GPe (globus pallidus externa) form a
feedback loop at the heart of basal ganglia dynamics.  In Parkinson's
disease this loop locks into strong, synchronised beta-band (~13-30 Hz)
oscillations.

Regime is controlled entirely by g_GABA, the strength of GPe→STN
inhibitory coupling:
  - healthy:      g_GABA = 0.3  → neurons fire tonically and independently
  - pathological: g_GABA = 1.5  → neurons synchronise into beta bursts

All time values are in milliseconds (ms).
All voltages are in millivolts (mV).
Conductances are in mS/cm².  Currents are in μA/cm².
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# State vector layout
# ---------------------------------------------------------------------------
# The ODE solver works on a single flat array y[].  We interleave state
# variables cell-by-cell so that all variables for STN cell 0 come first,
# then STN cell 1, ..., then all GPe cells in the same order.
#
# For 1 STN cell + 1 GPe cell (MVP) the layout is:
#
#   index:  0    1    2    3    4       5        6    7    8    9    10      11
#   var:  [V_s, h_s, n_s, r_s, Ca_s, s_AMPA,  V_g, h_g, n_g, r_g, Ca_g, s_GABA]
#          ^--- STN cell 0 ----------------------^  ^--- GPe cell 0 -------------^
#
# Variables:
#   V     - membrane voltage (mV)
#   h     - sodium channel inactivation gate (dimensionless, 0-1)
#   n     - potassium channel activation gate (dimensionless, 0-1)
#   r     - T-type calcium channel inactivation gate (dimensionless, 0-1)
#   Ca    - intracellular calcium concentration (mM)
#   s_AMPA  - STN presynaptic release gate; drives GPe (dimensionless, 0-1)
#   s_GABA  - GPe presynaptic release gate; drives STN (dimensionless, 0-1)

STN_STATE_DIM: int = 6   # number of state variables per STN cell
GPE_STATE_DIM: int = 6   # number of state variables per GPe cell


# ---------------------------------------------------------------------------
# STN neuron parameters
# ---------------------------------------------------------------------------

@dataclass
class STNParams:
    """
    Biophysical parameters for a single STN neuron.

    Based on Terman & Rubin (2002) "Activity patterns in a model for the
    subthalamopallidal network of the basal ganglia."

    The STN neuron has five ionic currents:
      I_L   - leak (passive, always-open channels)
      I_Na  - fast transient sodium  (causes the rising phase of an action potential)
      I_K   - delayed-rectifier potassium  (repolarises the cell after a spike)
      I_T   - low-threshold T-type calcium  (enables burst firing after hyperpolarisation)
      I_Ca  - high-threshold calcium  (contributes to plateau potentials)
      I_AHP - calcium-activated potassium  (slow after-hyperpolarisation, sets firing rate)
    """

    # --- Membrane capacitance ---
    C: float = 1.0          # μF/cm²  — scales how fast voltage changes

    # --- Leak current (I_L = g_L * (V - E_L)) ---
    g_L: float = 2.25       # mS/cm²  — conductance of always-open leak channels
    E_L: float = -60.0      # mV      — reversal potential (resting voltage baseline)

    # --- Sodium current (I_Na = g_Na * m_inf(V)^3 * h * (V - E_Na)) ---
    # m is the activation gate (so fast it is treated as instantaneous → m_inf)
    # h is the inactivation gate (slow; recovers during hyperpolarisation)
    g_Na: float = 37.5      # mS/cm²
    E_Na: float = 55.0      # mV  — sodium wants to drive V up to +55 mV

    # --- Potassium current (I_K = g_K * n^4 * (V - E_K)) ---
    # n is the delayed activation gate (opens after the spike peak)
    g_K: float = 45.0       # mS/cm²
    E_K: float = -80.0      # mV  — potassium wants to pull V down to -80 mV

    # --- T-type calcium current (I_T = g_T * a_inf(V)^3 * r * (V - E_Ca)) ---
    # Low-threshold burst current.  a is instantaneous activation,
    # r is a slow inactivation gate that recovers during hyperpolarisation.
    # Responsible for rebound burst firing after inhibition.
    g_T: float = 0.5        # mS/cm²

    # --- High-threshold calcium current (I_Ca = g_Ca * s_inf(V)^2 * (V - E_Ca)) ---
    # s is instantaneous activation.  Contributes to plateau depolarisations.
    g_Ca: float = 0.5       # mS/cm²
    E_Ca: float = 140.0     # mV  — shared reversal for both Ca currents

    # --- AHP potassium current (I_AHP = g_AHP * (Ca/(Ca+k1)) * (V - E_K)) ---
    # Activated by intracellular calcium; slows firing rate after bursts.
    g_AHP: float = 9.0      # mS/cm²
    k1: float = 15.0        # mM  — half-activation Ca concentration

    # --- Intracellular calcium dynamics ---
    # dCa/dt = eps * (-I_Ca - I_T - k_Ca * Ca)
    # eps scales how much current flow changes [Ca]
    # k_Ca is the Ca extrusion / buffering rate
    eps: float = 3.75e-5    # mM / (μA/cm² · ms)
    k_Ca: float = 22.5      # ms⁻¹

    # --- Tonic applied current ---
    # In isolation the STN neuron fires tonically at ~25 Hz when I_app = 33.
    # This represents excitatory drive from cortex / thalamus.
    I_app: float = 33.0     # μA/cm²


# ---------------------------------------------------------------------------
# GPe neuron parameters
# ---------------------------------------------------------------------------

@dataclass
class GPeParams:
    """
    Biophysical parameters for a single GPe neuron.

    GPe neurons are similar in structure to STN neurons but have
    different conductance magnitudes.  GPe fires faster and is more
    excitable.  Key difference: much larger I_AHP (g_AHP=30 vs 9),
    making them prone to rhythmic burst-pause patterns.
    """

    C: float = 1.0          # μF/cm²

    # Leak
    g_L: float = 0.1        # mS/cm²  — much smaller leak than STN
    E_L: float = -65.0      # mV

    # Sodium
    g_Na: float = 120.0     # mS/cm²  — larger Na conductance → faster spikes
    E_Na: float = 55.0      # mV

    # Potassium
    g_K: float = 30.0       # mS/cm²
    E_K: float = -80.0      # mV

    # T-type calcium
    g_T: float = 0.5        # mS/cm²

    # High-threshold calcium
    g_Ca: float = 0.1       # mS/cm²  — smaller than STN
    E_Ca: float = 120.0     # mV  — slightly lower than STN

    # AHP
    g_AHP: float = 30.0     # mS/cm²  — 3× larger than STN → stronger rate adaptation
    k1: float = 30.0        # mM      — 2× larger → less sensitive to Ca

    # Calcium dynamics
    eps: float = 1e-4       # faster Ca response than STN
    k_Ca: float = 20.0      # ms⁻¹

    # Tonic applied current (drive from striatum / other inputs)
    I_app: float = 20.0     # μA/cm²


# ---------------------------------------------------------------------------
# Synaptic coupling parameters
# ---------------------------------------------------------------------------

@dataclass
class SynapticParams:
    """
    Parameters governing the synaptic connections between STN and GPe.

    Synapse model: first-order kinetics with a smooth Heaviside trigger.
        ds/dt = alpha * (1 - s) * H(V_pre) - beta * s
    where H(V) = 1/(1 + exp(-(V - theta_syn)/sigma_syn)) approximates
    the probability of transmitter release as a function of presynaptic
    voltage.  s then gates the postsynaptic current:
        I_syn = g_weight * s * (V_post - E_rev)

    STN→GPe uses an excitatory (AMPA-like) synapse.
    GPe→STN uses an inhibitory (GABA-A-like) synapse.
    The actual GPe→STN conductance weight is set via SimConfig.g_GABA
    so the regime can be switched without touching this dataclass.
    """

    # ---- STN→GPe excitatory synapse (AMPA-like) ----
    g_AMPA: float = 0.3         # mS/cm²  — peak conductance weight
    E_AMPA: float = 0.0         # mV      — excitatory reversal potential
    alpha_AMPA: float = 5.0     # ms⁻¹   — fast opening rate
    beta_AMPA: float = 1.0      # ms⁻¹   — closing rate
    theta_syn_s: float = -20.0  # mV      — STN voltage at 50% release probability
    sigma_syn_s: float = 0.4    # mV      — steepness; small = near-digital switch

    # ---- GPe→STN inhibitory synapse (GABA-A-like) ----
    # Conductance weight is in SimConfig.g_GABA (regime-dependent)
    E_GABA: float = -85.0       # mV      — inhibitory reversal; below E_K → shunting
    alpha_GABA: float = 2.0     # ms⁻¹
    beta_GABA: float = 0.08     # ms⁻¹   — slow closing → longer IPSPs than AMPA
    theta_syn_g: float = -20.0  # mV      — GPe voltage at 50% release probability
    sigma_syn_g: float = 0.4    # mV


# ---------------------------------------------------------------------------
# Top-level simulation configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """
    Everything needed to run one simulation trajectory.

    The single most important field is g_GABA — it determines whether
    the network produces healthy tonic firing or pathological beta bursts.
    Use healthy_config() / pathological_config() to get pre-set versions.
    """

    # Number of cells in each population.
    # 8+8 is the default: large enough for the population mean-field averaging
    # that drives beta synchrony in Terman-Rubin, while staying fast to solve.
    # The 1+1 single-cell topology failed the sanity gate — with only one
    # GPe→STN connection the effective inhibitory drive is ~16× too weak to
    # trigger the rebound synchrony loop at published parameter values.
    n_STN: int = 8
    n_GPe: int = 8

    # Neuron-level parameters (one shared parameter set per population)
    stn: STNParams = field(default_factory=STNParams)
    gpe: GPeParams = field(default_factory=GPeParams)
    syn: SynapticParams = field(default_factory=SynapticParams)

    # ---- Primary regime knob ----
    # GPe→STN inhibitory peak conductance (mS/cm²).
    # Low  → GPe cannot strongly inhibit STN → STN fires tonically (healthy).
    # High → GPe strongly inhibits STN, rebounds, fires again → sync loop (pathological).
    g_GABA: float = 0.3

    # ---- Simulation timing (ms) ----
    t_start: float = 0.0
    t_end: float = 3000.0     # total run duration (ms); 3 s default
    t_warmup: float = 500.0   # ms to discard at the start as transient
                               # (network takes ~200-500 ms to reach steady state)

    # ---- ODE solver settings ----
    dt_max: float = 0.1       # maximum allowed step size (ms)
                               # action potentials last ~1-2 ms so 0.1 ms captures them

    # ---- Output resampling ----
    fs: float = 250.0         # target sample rate (Hz) — locked by experiment contract
                               # 250 Hz → 1 sample every 4 ms


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def healthy_config(**overrides) -> SimConfig:
    """
    Return a SimConfig configured for the healthy (asynchronous) regime.

    g_GABA = 0.3 → GPe inhibition is too weak to entrain STN into
    synchronised bursting.  Both populations fire tonically.
    """
    cfg = SimConfig(**overrides)
    cfg.g_GABA = 0.3
    return cfg


def pathological_config(**overrides) -> SimConfig:
    """
    Return a SimConfig configured for the pathological (beta-synchronised) regime.

    g_GABA = 1.5 → Strong GPe→STN inhibition creates a rebound excitation
    loop at beta-band frequencies (~13-30 Hz), mimicking Parkinsonian LFP.
    """
    cfg = SimConfig(**overrides)
    cfg.g_GABA = 1.5
    return cfg
