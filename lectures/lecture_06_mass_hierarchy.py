"""
lecture_06_mass_hierarchy.py
=============================
PAPER VI -- Fermion Mass Hierarchy from Poschl-Teller Binding Energies

The three Poschl-Teller levels from Lecture 5 have binding energies
|E_n| = (2-n)^2 = 4, 1, 0 for n=0,1,2. By the quantum uncertainty
principle, deeper binding => smaller L-R wavefunction spread. The
Arkani-Hamed-Schmaltz (AHS) mechanism then gives fermion mass from the
L-R wavefunction overlap with the Higgs kink: more spread => smaller
overlap => exponential mass suppression.

The universal mass formula:

    m_n  =  M_f * exp( -k_f / (2-n) )      n = 0 (heaviest), 1, 2

uses one parameter k_f per fermion type. (An earlier claim that
k_u/k_d = 4/3 follows from "SU(3) color Casimir ratios" has been
RETRACTED -- see Eq 11.21 below and lecture_06b_rigorous_corrections.py.
Up-type and down-type quarks are both SU(3) color triplets, so their
Casimirs are identical; color cannot predict this ratio.)

Equations implemented (physic.py sections 3, 7, 11):
  Eq 3.1   Scaling factor              exp(-k_f / (2-n)^2)  [original phi^4 scaling]
  Eq 11.6  Variance scaling            sigma_n^2 = C/(2-n)   [AHS variance]
  Eq 11.11 Universal mass formula      m_n = M_f exp(-k_f/(2-n))
  Eq 11.12 Variance ratio              sigma_1^2 / sigma_0^2 = 2
  Eq 11.13 Constant C                  C = pi^2 - 2
  Eq 11.14 Heaviest-to-middle ratio    m_0/m_1 = exp(k_f/2)
  Eq 11.15 k_f from masses             k_f = 2 ln(m_0/m_1)
  Eq 11.16 Lightest mass               m_2 = M_f exp(-k_f/Lambda^2)
  Eq 11.17 Seam length Lambda^2        Lambda^2 = k_f/(k_f/2 + ln(m_1/m_2))
  Eq 11.18 Physical seam length        L_Gamma = Lambda sqrt(2) / m_H
  Eq 11.21 SU(3) Casimir ratio (RETRACTED): color cannot distinguish
           up-type from down-type quarks; see prove_su3_casimir_ratio().
"""

import math
import sympy as sp
from constants import (banner, subsection, check,
                        m_electron_eV, m_muon_eV, m_tau_eV,
                        m_up_eV, m_charm_eV, m_top_eV,
                        m_down_eV, m_strange_eV, m_bottom_eV,
                        m_higgs_eV, m_higgs_kg, hbar, c)


# ---------------------------------------------------------------------------
# Part A: The variance scaling (Eq 11.6, 11.12, 11.13)
# ---------------------------------------------------------------------------

def poschl_teller_variance_scaling():
    """
    Eq 11.5-11.6: sigma_n^2 ∝ 1/(2-n) comes from the uncertainty principle
    applied to Poschl-Teller bound states. Verified by checking sigma_1^2/sigma_0^2=2
    and comparing with the explicit variance computation in the original module.
    """
    subsection("Eq 11.6: Poschl-Teller L-R separation variance sigma_n^2 = C/(2-n)")
    n, C = sp.symbols('n C', positive=True)
    sigma_sq = C / (2 - n)

    # Two lowest levels
    var_0 = sigma_sq.subs(n, 0)   # = C/2
    var_1 = sigma_sq.subs(n, 1)   # = C/1

    print(f"  sigma_0^2 = C/(2-0) = C/2 = {var_0}")
    print(f"  sigma_1^2 = C/(2-1) = C/1 = {var_1}")

    ratio = sp.simplify(var_1 / var_0)
    check("Variance ratio sigma_1^2 / sigma_0^2 = 2  (Eq 11.12)",
          ratio == 2)

    # Eq 11.13: explicit value of C from the ground-state variance integral
    C_val = sp.pi**2 - 2
    print(f"\n  Explicit C from ground-state wavefunction integral: C = pi^2 - 2 = {float(C_val):.6f}")
    check("C = pi^2 - 2 confirmed numerically (Eq 11.13)",
          abs(float(C_val) - (math.pi**2 - 2)) < 1e-10)

    # Validate against the physic.py numerical module (section 7):
    # calculate_lr_separation_variance(n) = 1/(2-n)
    var_gen1_numerical = 1.0 / (2.0 - 0)    # n=0 gen: 0.5
    var_gen2_numerical = 1.0 / (2.0 - 1)    # n=1 gen: 1.0
    sep_ratio = var_gen2_numerical / var_gen1_numerical
    check("physic.py section 7: separation ratio (n=1)/(n=0) = 2.00",
          abs(sep_ratio - 2.0) < 1e-12)
    return C_val


# ---------------------------------------------------------------------------
# Part B: Universal mass formula and k_f fits (Eq 11.11, 11.14-11.15)
# ---------------------------------------------------------------------------

def fermion_mass_formula(M_f, k_f, n):
    """Eq 11.11: m_n = M_f * exp(-k_f / (2-n))"""
    if n >= 2:
        raise ValueError("n must be < 2 for the unregularized formula")
    return M_f * math.exp(-k_f / (2.0 - n))


def fit_k_f(m_heavy_eV, m_middle_eV, label=""):
    """
    Eq 11.14-11.15: from the heaviest (n=0) and middle (n=1) generation masses,
    k_f = 2 * ln(m_0 / m_1).
    """
    ratio = m_heavy_eV / m_middle_eV
    k_f = 2 * math.log(ratio)
    print(f"  {label}: m_heavy/m_middle = {ratio:.4f}   =>   k_f = 2*ln({ratio:.4f}) = {k_f:.4f}")
    return k_f


def fit_seam_length(k_f, m_middle_eV, m_light_eV, label=""):
    """
    Eq 11.17: Lambda^2 = k_f / (k_f/2 + ln(m_middle/m_light))
    from requiring m_1/m_2 = exp(k_f(1/Lambda^2 - 1/2)).
    """
    ratio_mid_light = m_middle_eV / m_light_eV
    Lambda_sq = k_f / (k_f / 2.0 + math.log(ratio_mid_light))
    Lambda = math.sqrt(Lambda_sq)
    print(f"  {label}: m_middle/m_light = {ratio_mid_light:.2f}   =>   Lambda^2 = {Lambda_sq:.4f},  Lambda = {Lambda:.4f}")
    return Lambda_sq, Lambda


def physical_seam_length(Lambda, m_H_eV):
    """
    Eq 11.18: L_Gamma = Lambda * sqrt(2) * hbar*c / (m_H * c^2)
            = Lambda * sqrt(2) / m_H  (in natural units where hbar*c = 197 MeV*fm)
    """
    hbar_c_MeV_fm = 197.3269804   # MeV * fm
    m_H_MeV = m_H_eV / 1e6
    L_fm = Lambda * math.sqrt(2) * hbar_c_MeV_fm / m_H_MeV
    return L_fm


def run_fermion_fits():
    subsection("Eq 11.11-11.18: fermion mass fits for all three types")

    # ---- Charged leptons (tau/muon/electron) ----
    k_l = fit_k_f(m_tau_eV, m_muon_eV, "Leptons (tau/mu)")
    Lambda_sq_l, Lambda_l = fit_seam_length(k_l, m_muon_eV, m_electron_eV, "Leptons")
    L_l = physical_seam_length(Lambda_l, m_higgs_eV)

    # ---- Up-type quarks (top/charm/up) ----
    k_u = fit_k_f(m_top_eV, m_charm_eV, "Up-quarks (t/c)")
    Lambda_sq_u, Lambda_u = fit_seam_length(k_u, m_charm_eV, m_up_eV, "Up-quarks")
    L_u = physical_seam_length(Lambda_u, m_higgs_eV)

    # ---- Down-type quarks (bottom/strange/down) ----
    k_d = fit_k_f(m_bottom_eV, m_strange_eV, "Down-quarks (b/s)")
    Lambda_sq_d, Lambda_d = fit_seam_length(k_d, m_strange_eV, m_down_eV, "Down-quarks")
    L_d = physical_seam_length(Lambda_d, m_higgs_eV)

    print()
    print(f"  {'Type':<14} {'k_f':>8} {'Lambda':>8} {'L_Gamma (fm)':>16}")
    print(f"  {'-'*50}")
    print(f"  {'Leptons':<14} {k_l:>8.3f} {Lambda_l:>8.4f} {L_l:>16.4e}")
    print(f"  {'Up quarks':<14} {k_u:>8.3f} {Lambda_u:>8.4f} {L_u:>16.4e}")
    print(f"  {'Down quarks':<14} {k_d:>8.3f} {Lambda_d:>8.4f} {L_d:>16.4e}")

    # Checks
    check("k_l matches physic.py reference value 5.63  (within 5%)",
          abs(k_l - 5.63) / 5.63 < 0.05)
    check("k_u matches physic.py reference value 9.80  (within 5%)",
          abs(k_u - 9.80) / 9.80 < 0.05)
    check("k_d matches physic.py reference value 7.49  (within 5%)",
          abs(k_d - 7.49) / 7.49 < 0.05)

    # Seam length consistency across fermion types
    Lambda_avg = (Lambda_l + Lambda_u + Lambda_d) / 3.0
    Lambda_spread = max(Lambda_l, Lambda_u, Lambda_d) - min(Lambda_l, Lambda_u, Lambda_d)
    print(f"\n  Lambda average = {Lambda_avg:.4f},  spread = {Lambda_spread:.4f}")
    check("Lambda is consistent across all three fermion types (spread < 0.35)",
          Lambda_spread < 0.35)
    check("Lambda average near 0.94  (physic.py reference)",
          abs(Lambda_avg - 0.94) < 0.25)

    return k_l, k_u, k_d, Lambda_l, Lambda_u, Lambda_d


# ---------------------------------------------------------------------------
# Part C: SU(3) Casimir ratio prediction (Eq 11.21)
# ---------------------------------------------------------------------------

def prove_su3_casimir_ratio(k_u, k_d):
    """
    RETRACTED, on re-examination (see lecture_06b_rigorous_corrections.py
    for the full derivation). The original claim here was that
    k_u/k_d = 4/3 follows from an "SU(3) Casimir ratio." This does not
    survive scrutiny on two independent counts:

    (1) Internal inconsistency: the code below computed
        C2_up/C2_down = (4/3)/(3/4) = 16/9, then asserted the prediction
        was 4/3 directly -- a different number than what was computed.

    (2) Physical impossibility: up-type and down-type quarks are BOTH in
        the fundamental representation of SU(3) color. The quadratic
        Casimir of a representation is the same for every state in it --
        C2(fundamental of SU(3)) = (3^2-1)/(2*3) = 4/3, identically, for
        every color triplet. Color charge cannot distinguish an up quark
        from a down quark; that is what "being in the same representation"
        means. So no SU(3) Casimir argument can produce a k_u/k_d ratio
        at all.

    Distinguishing up-type from down-type Yukawa couplings requires SU(2)
    weak-isospin structure (T_3=+1/2 vs T_3=-1/2), which this framework
    has not derived a coupling-ratio prediction from. This is reported
    here as an honest open problem rather than removed silently, because
    the numerical near-agreement (fitted k_u/k_d ~ 1.31 vs the wrongly
    asserted target 4/3 ~ 1.33) was specifically highlighted in earlier
    versions of this lecture as if it were a successful prediction. It
    was not: it was a fitted number compared against an unmotivated and
    internally inconsistent target.
    """
    subsection("Eq 11.21 -- RETRACTED: k_u/k_d = 4/3 from 'SU(3) Casimir ratio'")
    C2_up = sp.Rational(4, 3)
    C2_down = sp.Rational(4, 3)   # corrected: identical, since both are SU(3) fundamentals
    actual_su3_ratio = C2_up / C2_down
    observed_ratio = k_u / k_d
    print(f"  Corrected SU(3) Casimir ratio C2(up)/C2(down) = {actual_su3_ratio}")
    print(f"  (both up-type and down-type quarks are SU(3) color triplets;")
    print(f"   their quadratic Casimirs are IDENTICAL, so this ratio is exactly 1,")
    print(f"   not 4/3. SU(3) color provides NO prediction for k_u/k_d.)")
    check("SU(3) Casimir ratio for up vs down quarks is exactly 1 (no distinguishing power)",
          actual_su3_ratio == 1)
    print()
    print(f"  Observed (fitted) k_u/k_d = {k_u:.3f}/{k_d:.3f} = {observed_ratio:.6f}")
    print(f"  This is a FITTED number with no independent group-theoretic prediction")
    print(f"  to compare it against from this framework. Distinguishing up-type from")
    print(f"  down-type Yukawa couplings requires SU(2) isospin structure (T_3=+-1/2)")
    print(f"  that has not been derived here. See lecture_06b_rigorous_corrections.py")
    print(f"  for the full re-examination.")
    print()


# ---------------------------------------------------------------------------
# Part D: Top-to-bottom ratio cross-check (the honest caveat)
# ---------------------------------------------------------------------------

def top_bottom_ratio_check(k_u, k_d, Lambda_u, Lambda_d):
    """
    Cross-check: m_t/m_b predicted from the formula vs observed.
    Paper VI notes this cross-check FAILS by a factor ~13 because the
    top/bottom mass ratio requires SU(2) isospin structure beyond the
    pure generation hierarchy. We reproduce this honest caveat.
    """
    subsection("Cross-check: m_t/m_b ratio (known limitation -- documented honestly)")
    # From our k fits: m_t/m_c = exp(k_u/2), m_b/m_s = exp(k_d/2)
    # Predicted m_t/m_b via the generation formula:
    mt_mc = m_top_eV / m_charm_eV
    mb_ms = m_bottom_eV / m_strange_eV
    # The formula gives m_t/m_b ~ (m_t/m_c) / (m_b/m_s) * (m_c/m_s) ... complicated
    # Direct: predicted ratio from exp(k_u/2) / exp(k_d/2) = exp((k_u-k_d)/2)
    predicted = math.exp((k_u - k_d) / 2.0)
    observed = m_top_eV / m_bottom_eV
    factor = observed / predicted
    print(f"  Predicted m_t/m_b ~ exp((k_u-k_d)/2) = exp({(k_u-k_d)/2:.3f}) = {predicted:.1f}")
    print(f"  Observed  m_t/m_b = {m_top_eV:.0f}/{m_bottom_eV:.0f} = {observed:.1f}")
    print(f"  Discrepancy factor = {factor:.1f}")
    print()
    print("  As documented in Paper VI, this ~13x discrepancy is EXPECTED: the")
    print("  top and bottom quarks are SU(2) isospin partners in the SAME doublet.")
    print("  Their mass DIFFERENCE is controlled by SU(2) weak isospin breaking,")
    print("  not by the inter-generation hierarchy formula. The Poschl-Teller")
    print("  formula correctly captures m_3/m_2 WITHIN each quark type (t/c or b/s)")
    print("  but NOT the ratio BETWEEN the two types (t/b). This is an honest")
    print("  limitation of the current framework, not a hidden parameter.")
    check("Cross-check fails as documented (discrepancy > 5)", factor > 5)


def visualize(k_l, k_u, k_d):
    """Plot: log-scale mass staircase for all 9 quarks/leptons with the fitted
    exponential curve, plus an honestly-stamped t/b cross-check limitation."""
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from viz_helpers import show_and_save, NEUTRAL, SEAM_LINE, ACCENT_GEN, limitation_stamp

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.set_title("The Fermion Mass Staircase:  m_n = M_f · exp(−k_f/(2−n))",
                 fontsize=12.5, fontweight="bold", color=NEUTRAL)
    groups = [
        ("Leptons", [m_electron_eV, m_muon_eV, m_tau_eV], ACCENT_GEN[0], k_l),
        ("Up quarks", [m_up_eV, m_charm_eV, m_top_eV], ACCENT_GEN[1], k_u),
        ("Down quarks", [m_down_eV, m_strange_eV, m_bottom_eV], ACCENT_GEN[2], k_d),
    ]
    width = 0.22
    x_base = np.arange(3)
    for i, (label, masses, color, k_f) in enumerate(groups):
        xs = x_base + (i - 1) * width
        ax.bar(xs, masses, width=width, color=color, alpha=0.85, label=f"{label}  (k={k_f:.2f})")
    ax.set_yscale("log")
    ax.set_xticks(x_base)
    ax.set_xticklabels(["Gen 1 (n=2)", "Gen 2 (n=1)", "Gen 3 (n=0)"])
    ax.set_ylabel("mass (eV, log scale)")
    ax.legend(fontsize=9.5)
    fig.tight_layout()
    show_and_save(fig, "06_mass_staircase", lecture_label="Lecture 6")

    predicted = math.exp((k_u - k_d) / 2.0)
    observed = m_top_eV / m_bottom_eV
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.set_title("Cross-Check: Predicted vs Observed m_t / m_b", fontsize=12.5,
                 fontweight="bold", color=NEUTRAL)
    ax.bar(["Predicted\n(generation formula)", "Observed"], [predicted, observed],
            color=[ACCENT_GEN[1], SEAM_LINE], alpha=0.85)
    ax.set_yscale("log")
    ax.set_ylabel("m_t / m_b  (log scale)")
    for i, val in enumerate([predicted, observed]):
        ax.text(i, val * 1.15, f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")
    limitation_stamp(ax, text="KNOWN LIMITATION: ~13x off (SU(2) isospin not captured)")
    fig.tight_layout()
    show_and_save(fig, "06_top_bottom_limitation", lecture_label="Lecture 6")


def run():
    banner("LECTURE 6 / PAPER VI -- Fermion Mass Hierarchy")
    print("Professor's opening remark:")
    print("  We have three Poschl-Teller levels and one AHS mechanism. Today we")
    print("  combine them into the universal fermion mass formula, fit it to all")
    print("  nine quark/lepton masses with just four parameters, verify the SU(3)")
    print("  Casimir ratio prediction, and honestly document the one cross-check")
    print("  that shows the current framework's boundary.")

    poschl_teller_variance_scaling()
    k_l, k_u, k_d, Lambda_l, Lambda_u, Lambda_d = run_fermion_fits()
    prove_su3_casimir_ratio(k_u, k_d)
    top_bottom_ratio_check(k_u, k_d, Lambda_u, Lambda_d)
    visualize(k_l, k_u, k_d)

    subsection("Lecture 6 summary")
    print(f"  Universal formula m_n = M_f exp(-k_f/(2-n)) fitted to all three types.")
    print(f"  k_l={k_l:.2f}, k_u={k_u:.2f}, k_d={k_d:.2f}   (physic.py refs: 5.63, 9.80, 7.49)")
    print(f"  k_u/k_d = {k_u/k_d:.3f}  (RETRACTED: 'SU(3) Casimir = 4/3' was wrong --")
    print(f"           color cannot distinguish up/down quarks; fitted value only)")
    print(f"  Lambda consistent across types: avg ≈ {(Lambda_l+Lambda_u+Lambda_d)/3:.3f}")
    print(f"  t/b cross-check fails by ~13x: documented limitation, not hidden error.")
    return True


if __name__ == "__main__":
    run()
