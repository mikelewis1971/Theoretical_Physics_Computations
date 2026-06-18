"""
constants.py
============
Shared physical constants and particle data used throughout the masterclass.
All masses are quoted from the Particle Data Group (PDG 2024) central values.
"""

import scipy.constants as const

# ---------------------------------------------------------------------------
# Fundamental constants (SI units)
# ---------------------------------------------------------------------------
c = const.c            # speed of light, m/s
hbar = const.hbar       # reduced Planck constant, J*s
h_planck = const.h      # Planck constant, J*s
eV = const.eV           # 1 eV in joules
G_NEWTON = const.G      # Newton's gravitational constant, m^3/(kg s^2)

# ---------------------------------------------------------------------------
# Particle masses, eV/c^2 (PDG central values)
# ---------------------------------------------------------------------------
m_electron_eV = 0.51099895e6
m_muon_eV = 105.658375e6
m_tau_eV = 1776.86e6

m_up_eV = 2.16e6
m_charm_eV = 1270e6
m_top_eV = 172570e6

m_down_eV = 4.67e6
m_strange_eV = 93.4e6
m_bottom_eV = 4180e6

m_higgs_eV = 125.20e9
m_planck_eV = 1.22089e28  # ~1.22 x 10^19 GeV

# ---------------------------------------------------------------------------
# Convenience conversions: eV/c^2 -> kg
# ---------------------------------------------------------------------------
def ev_to_kg(m_eV):
    """Convert a mass given in eV/c^2 to kilograms."""
    return m_eV * eV / c ** 2

m_electron_kg = ev_to_kg(m_electron_eV)
m_muon_kg = ev_to_kg(m_muon_eV)
m_tau_kg = ev_to_kg(m_tau_eV)
m_higgs_kg = ev_to_kg(m_higgs_eV)
m_planck_kg = ev_to_kg(m_planck_eV)

# ---------------------------------------------------------------------------
# Observed mixing / cosmological inputs used for cross-checks
# ---------------------------------------------------------------------------
CKM_DELTA_RAD = 1.20          # CP-violating phase, radians (PDG fit, ~68.8 deg)
LAMBDA_OBS_INV_M2 = 1.1e-52   # observed cosmological constant, m^-2

# ---------------------------------------------------------------------------
# Pretty-printing helpers used by every lecture module
# ---------------------------------------------------------------------------
RULE = "=" * 78
THIN = "-" * 78


def banner(title):
    print("\n" + RULE)
    print(f" {title}")
    print(RULE)


def subsection(title):
    print("\n" + THIN)
    print(f" {title}")
    print(THIN)


def check(label, condition):
    """Print a pass/fail line for a boolean proof result."""
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}")
    return condition
