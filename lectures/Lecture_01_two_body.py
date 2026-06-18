"""
lecture_01_two_body.py
=======================
PAPER I -- The Two-Body Heisenberg Algebra and the Pair Axis

Every result in this masterclass traces back to one elementary fact about
two-particle quantum mechanics: when you switch from individual particle
variables to collective (center-of-mass / relative) variables, two of the
four canonical commutators vanish identically. That is the seed of the
entire chiral-seam framework.

Equations implemented:
  Eq 1.1   [X, P]   = i hbar      (collective canonical pair, standard)
  Eq 1.2   [rho, p] = i hbar      (relative canonical pair, standard)
  Eq 1.3   [rho, P] = 0           (the KEY vanishing commutator)
  Eq 1.4   [X, p]   = 0           (the second vanishing commutator)
  Eq 1.5   rho = x_L - x_R        (relative coordinate definition)
  Eq 1.6   Pair-axis construction from P=0 kinematics (geodesic argument)
"""

import sympy as sp
from constants import banner, subsection, check


def build_two_body_operators():
    """
    Builds explicit (non-commutative) SymPy symbols for the four collective
    phase-space operators of a two-body system:
        X   = center-of-mass position
        P   = total momentum
        rho = relative separation  (x_L - x_R)
        p   = relative momentum
    and the matching individual-particle operators x1, p1, x2, p2 from which
    they are constructed, so that the vanishing commutators are *derived*
    rather than merely asserted.
    """
    x1, p1, x2, p2, hbar = sp.symbols('x1 p1 x2 p2 hbar', commutative=False)
    return x1, p1, x2, p2, hbar


def derive_collective_commutators():
    """
    Eq 1.1-1.4.

    Starts from the only assumption of single-particle quantum mechanics:
        [x_i, p_j] = i hbar delta_ij     (standard, particles independent)
    and defines the collective variables
        X   = (x1 + x2) / 2
        P   = p1 + p2
        rho = x1 - x2
        p   = (p1 - p2) / 2
    then computes all four cross commutators symbolically and proves that
    [rho, P] = 0 and [X, p] = 0 *follow* from the single-particle algebra --
    they are not extra postulates.
    """
    subsection("Deriving the collective commutators from single-particle QM")

    # Treat x1,p1,x2,p2 as abstract non-commuting generators whose only
    # relations are the canonical ones. We implement the commutator algebra
    # by hand (since sympy's noncommutative symbols don't know [x,p]=ihbar by
    # default), using a tiny rewriting rule: anything of the form
    # p_i*x_i - x_i*p_i is replaced by -i*hbar (and x_i*p_i - p_i*x_i by +i*hbar),
    # while operators with different particle indices always commute.
    hbar = sp.Symbol('hbar', positive=True)
    I = sp.I

    # We track each collective operator as a formal linear combination of the
    # four single-particle operators, then compute the commutator using the
    # *bilinearity* of the commutator and the canonical relations:
    #   [x1,p1] = i hbar ,  [x2,p2] = i hbar ,  [x1,p2]=[x1,x2]=[p1,p2]=[x2,p1]=0
    #
    # We implement this with a tiny symbolic vector representation:
    #   operator = a*x1 + b*p1 + c*x2 + d*p2   (coefficients a,b,c,d)
    # and a bracket table for the four basis generators.

    basis = ['x1', 'p1', 'x2', 'p2']
    # bracket_table[u][v] = value of [u, v] in units of i*hbar (0 otherwise)
    bracket_table = {
        ('x1', 'p1'): 1, ('p1', 'x1'): -1,
        ('x2', 'p2'): 1, ('p2', 'x2'): -1,
    }

    def commutator(op1, op2):
        """op1, op2: dict {basis_label: coefficient}. Returns coefficient of i*hbar."""
        total = 0
        for u, cu in op1.items():
            for v, cv in op2.items():
                total += cu * cv * bracket_table.get((u, v), 0)
        return total

    # Collective operators as combinations of the basis
    X = {'x1': sp.Rational(1, 2), 'x2': sp.Rational(1, 2)}
    P = {'p1': 1, 'p2': 1}
    rho = {'x1': 1, 'x2': -1}
    p_rel = {'p1': sp.Rational(1, 2), 'p2': -sp.Rational(1, 2)}

    comm_X_P = commutator(X, P)
    comm_rho_p = commutator(rho, p_rel)
    comm_rho_P = commutator(rho, P)
    comm_X_p = commutator(X, p_rel)

    print("  Single-particle canonical relations assumed:")
    print("     [x1,p1] = i hbar ,  [x2,p2] = i hbar ,  all cross terms = 0")
    print()
    print(f"  [X, P]   = {comm_X_P} * i hbar   (Eq 1.1, standard Heisenberg pair)")
    print(f"  [rho, p] = {comm_rho_p} * i hbar   (Eq 1.2, standard Heisenberg pair)")
    print(f"  [rho, P] = {comm_rho_P} * i hbar   (Eq 1.3, THE KEY RESULT)")
    print(f"  [X, p]   = {comm_X_p} * i hbar   (Eq 1.4, the second key result)")

    ok1 = check("[X, P] = i hbar  (standard)", comm_X_P == 1)
    ok2 = check("[rho, p] = i hbar  (standard)", comm_rho_p == 1)
    ok3 = check("[rho, P] = 0  (relative separation commutes with TOTAL momentum)", comm_rho_P == 0)
    ok4 = check("[X, p] = 0  (center of mass commutes with RELATIVE momentum)", comm_X_p == 0)

    print()
    print("  Physical reading: the uncertainty principle is NOT violated. It is")
    print("  RELOCATED. [X,P] and [rho,p] still obey full Heisenberg uncertainty.")
    print("  But [rho,P]=0 and [X,p]=0 mean the pair's *separation* and its")
    print("  *total momentum* can be simultaneously sharp -- and that is exactly")
    print("  the situation in pair production with P_total = 0.")

    return all([ok1, ok2, ok3, ok4])


def derive_pair_axis_geodesic():
    """
    Eq 1.5-1.6.

    Physical realization: in a pair-production event (e.g. gamma -> e+ e-)
    in the center-of-momentum frame, P_total = 0 exactly (sharp), and by
    [rho, P] = 0 the relative separation rho is then free to also be sharp.
    Momentum conservation forces the two particles' momenta to be exactly
    antiparallel: p_1 = -p_2. Antipodal momentum directions define a unique
    great-circle axis on the momentum sphere -- a geodesic. That geodesic is
    the chiral seam of Paper II.
    """
    subsection("From P=0 pair production to a conserved geodesic axis")

    # Symbolic demonstration: two back-to-back 3-momenta of equal magnitude
    px, py, pz = sp.symbols('px py pz', real=True)
    p1_vec = sp.Matrix([px, py, pz])
    p2_vec = -p1_vec  # exact momentum conservation, P_total = 0

    total_p = p1_vec + p2_vec
    check("Total momentum P = p1 + p2 vanishes exactly", total_p == sp.zeros(3, 1))

    # The relative separation direction rho_hat is along p1_vec (by symmetry
    # of the two-body problem the line connecting the particles lies along
    # their common momentum axis in the CM frame).
    print("  p_1 = (px, py, pz),  p_2 = -p_1   (exact antiparallel kinematics)")
    print("  This pair of antipodal directions on the momentum sphere defines")
    print("  a unique great circle -- a geodesic. In Paper II this geodesic")
    print("  becomes the chiral seam: the boundary between the spherical and")
    print("  hyperbolic curvature domains.")
    return True


def rho_definition():
    """Eq 1.5: rho = x_L - x_R, the L/R separation used throughout Papers II, V, VI."""
    subsection("Definition: rho as the Left/Right separation")
    x_L, x_R = sp.symbols('x_L x_R', real=True)
    rho = x_L - x_R
    print(f"  rho := x_L - x_R = {rho}")
    print("  This is the SAME rho used in Papers II, V, and VI: the spatial")
    print("  separation between the left-handed and right-handed components")
    print("  of a single fermion is what oscillates as the particle crosses")
    print("  the seam, and it is what generates the mass hierarchy via the")
    print("  Arkani-Hamed-Schmaltz overlap mechanism (Lecture 6).")
    return rho


def run():
    banner("LECTURE 1 / PAPER I -- The Two-Body Heisenberg Algebra and the Pair Axis")
    print("Professor's opening remark:")
    print("  Before we can talk about geometry, gauge groups, or the Higgs, we")
    print("  need one fact from ordinary quantum mechanics: when a pair of")
    print("  particles is described by COLLECTIVE variables instead of")
    print("  individual ones, two commutators that 'should' be there vanish.")
    print("  That single algebraic fact is the seed of everything that follows.")

    ok_comm = derive_collective_commutators()
    derive_pair_axis_geodesic()
    rho_definition()

    subsection("Lecture 1 summary")
    print("  [rho, P] = 0  and  [X, p] = 0  are PROVED from single-particle QM.")
    print("  Physical pair production with P=0 fixes a conserved geodesic axis.")
    print("  That axis becomes the chiral seam in Lecture 2.")
    return ok_comm


if __name__ == "__main__":
    run()
