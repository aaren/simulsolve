import numpy as np
# import matplotlib as mpl
# mpl.use('Agg')
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import brentq, fsolve
from sympy.utilities.lambdify import lambdify

# define symbols used by sympy
# h - current depth
# U0 - front speed
# d1c - depth of first layer above current
# d0 - depth of undisturbed first layer (initial interface depth)
h, U0, d1c, d0 = sp.symbols('h, U0, d1c, d0')
# Cb - bore speed
# U11 - fluid velocity in first layer through bore
# U21 - fluid velocity in second layer through bore
# d11 - bore amplitude
# S - stratification parameter (rho1 - rho2) / (rhoc - rho2)
Cb, U11, U21, d11, S = sp.symbols('Cb, U11, U21, d11, S')


### BASE EQUATIONS ###
def eq27(h=h, U0=U0, d0=d0, d1c=d1c):
    """From the Bernoulli equation in two layer extra critical flow."""
    Lh = d0 - d1c - h
    Rh = (U0 ** 2 / 2) * ((d0 ** 2 / d1c ** 2)
                          - ((1 - d0) ** 2 / (1 - d1c - h) ** 2))
    f = Lh - Rh
    return f


def eq28(h=h, U0=U0, d0=d0, d1c=d1c, S=S):
    """Momentum conservation in two layer extra critical flow."""
    f = (h ** 2 / (2 * S)) - (h / S) + (d1c ** 2 / 2) - (d0 ** 2 / 2) \
        + d0 - d1c + (d1c * h) \
        + U0 ** 2 * (-.5 + (d0 ** 2 / d1c) + ((1 - d0) ** 2 / (1 - d1c - h)))
    return f


def eq29(h=h, S=S, U0=U0, d0=d0, d1c=d1c):
    f = h * (1 - S) / S - (U0 ** 2 / 2) * (d0 ** 2 / d1c ** 2)
    return f


def eq211(h=h, U0=U0, d1c=d1c, d0=d0):
    """white-helfrich2012 use h where they mean h/d0 - the scaling in
    Baines is w.r.t d0 rather than H. Corrected here."""
    f = (U0) ** 2 * ((d0 ** 2 / d1c ** 3) + (1 - d0) ** 2 / (1 - d1c - h) ** 3) - 1
    return f


def eq212(h=h, U0=U0, d1c=d1c, d0=d0):
    """NB. white-helfrich2012 has a sign error in eq212. corrected here.

    white-helfrich2012 use h where they mean h/d0 - the scaling in
    Baines is w.r.t d0 rather than H. Corrected here."""
    f = 0.5 * (U0) ** 2 * ((d0 ** 2 / d1c ** 2) - (1 - d0) ** 2 / (1 - d1c - h) ** 2) + d1c + (h - d0)
    return f


def eq31(U0=U0, Cb=Cb, d11=d11, d0=d0):
    """Klemp et al bore jump condition"""
    f = (U0 + Cb) ** 2 - ((d11 ** 2 * (1 - d11) * (2 - d11 - d0)) / (d11 + d0 + d11 ** 2 - 3 * d0 * d11))
    return f


def eq33(U11=U11, Cb=Cb, d11=d11, U0=U0, d0=d0):
    """Mass conservation #1"""
    f = (U11 + Cb) * d11 - (U0 + Cb) * d0
    return f


def eq34(U21=U21, Cb=Cb, d11=d11, U0=U0, d0=d0):
    """Mass conservation #2"""
    f = (U21 + Cb) * (1 - d11) - (U0 + Cb) * (1 - d0)
    return f


def eq35(d11=d11, d1c=d1c, h=h, U11=U11, U21=U21):
    """Bernoulli"""
    f = d11 - d1c - h - (U11 ** 2 / 2) * (d11 ** 2 / d1c ** 2 - 1) - (U21 ** 2 / 2) * (1 - (1 - d11) ** 2 / (1 - d1c - h) ** 2)
    return f


def eq36(h=h, S=S, d1c=d1c, d11=d11, U11=U11, U21=U21):
    """Momentum conservation. White & Helfrich 2012 contains an error.
    All d_0 should be replaced with d_11. Corrected here.
    """
    f = (h ** 2 / (2 * S)) - h / (S) + (d1c ** 2 / 2) - (d11 ** 2 / 2) \
        + d11 - d1c + d1c * h \
        + (U11 ** 2) * (0.5 + (d11 ** 2 / d1c) - d11) \
        + (U21 ** 2) * ((1 - d11) ** 2 / (1 - d1c - h) + d11 - 1)
    return f


def eq38(U0=U0, S=S, h=h, d11=d11, d1c=d1c, U11=U11, Dc=0):
    """Energy dissipation. Dc=0 gives rightward bound of resonant wedge."""
    f = h * ((1 - S) / S) - (U11 ** 2 / 2) * (d11 ** 2 / d1c ** 2) - Dc / U0
    return f


def eq39(U0=U0, d11=d11, U11=U11, U21=U21):
    """Bore criticality. Upper bound of resonant wedge."""
    # TODO: verify this
    f = U0 + (U11 - U21) * (1 - 2 * d11) - ((1 - (U11 - U21) ** 2) * d11 * (1 - d11)) ** .5
    return f


def longwave_c0(d0=d0):
    """Longwave speed given the interface depth d0"""
    return (d0 * (1 - d0)) ** .5

### /BASE EQUATIONS ###


### REARRANGEMENTS ###
def subbed():
    f1 = eq211()
    f2 = eq212()
    # rearrange f1 to get U0^2
    U2 = ((f1 + 1) / U0 ** 2) ** -1
    # substitute this into f2
    f3 = f2.subs(U0 ** 2, U2)
    return f3


def U(h=h, d0=d0, d1c=d1c):
    f1 = eq211(h=h, d0=d0, d1c=d1c)
    # rearrange f1 to get U0^2
    U2 = ((f1 + 1) / U0 ** 2) ** -1
    U = U2 ** .5
    return U


def U_subbed_poly():
    """Eliminate U_0 from eqs 2.11 and 2.12 and cancel out terms
    using the equality with zero.
    """
    f3 = subbed()
    # f3 is now f(h, d1c) only.
    # integer exponents --> rational fraction
    # so we only need to take the numerator of f3 as we are
    # looking for where it goes to 0. this is a polynomial in
    # (h, d1c, d0)
    print "constructing polynomial in d1c..."
    p3 = sp.fraction(f3.cancel())[0]
    return p3


def us():
    """Substitute out the bore speed to obtain relations
    for u11 and u12 as f(U0, d11, d0).
    """
    # TODO: check sign of roots
    # I think this is the correct root for Cb [1]
    # actually, taking the first one seems to get the right
    # answers...
    # maybe it makes no difference.
    cb = sp.solve(eq31(), Cb)[1]
    u11 = sp.solve(eq33().subs(Cb, cb), U11)[0]
    u21 = sp.solve(eq34().subs(Cb, cb), U21)[0]
    return u11, u21


def resonant_criterion():
    """Transforms eq 3.9 into a function of (d0, d11, U0)"""
    u11, u21 = us()
    crit = eq39(U11=u11, U21=u21)
    return crit
### /REARRANGEMENTS ###


### CRITICAL BOUNDS ###
def fast_solve(H, d0_=0.1, combined_poly=None):
    """Find the branches of solutions to two layer flow
    over topography for a given interface depth d0 and
    an array of topography heights H.

    Returns a list of arrays of U corresponding to H,
    with as many list elements as there are roots of the
    equations (4).
    """
    if not combined_poly:
        combined_poly = U_subbed_poly()
    # we want to get the coefficients, but also to be able to
    # evaluate over an input array. the latter requires that we
    # use lambdify.
    # we can do this!
    coeff = lambdify((d0, h), sp.Poly(combined_poly, d1c).coeffs(), "numpy")
    d0_ = np.array([d0_])
    coeffs = np.array(coeff(*np.meshgrid(d0_, H)))
    # now we have a list of arrays of coefficients, each array
    # having the dimension of H
    # np.roots can only take 1d input so we have to be explicit
    Roots = np.array([np.roots(coeffs[:, i].squeeze()) for i in range(len(H))])
    # Now we work out what U would be
    Uf = lambdify((d0, h, d1c), U(h=h, d0=d0, d1c=d1c), "numpy")
    U_sol = [Uf(d0_, H, Roots[:, i]) for i in range(Roots.shape[1])]
    return U_sol


def branch_select(U, H, d0):
    """Given a load of values of U and corresponding H,
    for a given d0, find the combinations of (U,H) that
    define the upper / lower limits of the critical flow
    region in two layer flow over topography.

    Inputs: U is an array of velocities.
            H is an array of heights that correspond to U.
            d0 is the interface depth (scalar)

    Returns a tuple (lower, upper) where lower and upper
    are tuples of the arrays of U, H that satisfy the
    criterion.

    The criterion for branch selection is 0 < h < d0 for
    both; 0 < U0 < c0 for the lower branch; c0 < U0 < 0.5
    for the upper branch.

    We use 0.5 as the upper limit as this is the speed of
    the conjugate state in two layers.

    c0 = ((1 - d0) * d0 ) ** .5 is the liner longwave speed
    on the interface.
    """
    # longwave speed
    c0 = longwave_c0(d0)

    cond_all = np.logical_and(0 < H, H < d0)
    cond_lower = np.logical_and(cond_all, np.logical_and(0 < U, U < c0))
    cond_upper = np.logical_and(cond_all, np.logical_and(c0 < U, U < 0.5))

    lower_branch = (U[cond_lower], H[cond_lower])
    upper_branch = (U[cond_upper], H[cond_upper])

    # ensure sorted w.r.t h
    p = lower_branch[1].argsort()
    q = upper_branch[1].argsort()
    sort_lower = (lower_branch[0][p], lower_branch[1][p])
    sort_upper = (upper_branch[0][q], upper_branch[1][q])

    return sort_lower, sort_upper


def critical_bounds(d0, H=None):
    if H is None:
        # h can be greater than d0 (current deeper than the
        # interface), but not for subcritical solutions. Deeper than
        # d0 corresponds to solutions in the critical or
        # supercritical regions.
        H = np.linspace(0, d0, 50)
    U_sol = fast_solve(H, d0)
    # Put all the U into a single array with a corresponding H array
    U = np.concatenate(U_sol)
    # H to match the U
    Hu = np.concatenate([H for Us in U_sol])
    lower, upper = branch_select(U, Hu, d0)
    return lower, upper

### /CRITICAL BOUNDS ###


### RIGHT BOUND ###
def f_subcrit(F, d0, d=0.01, bound=None):
    """Compute the intersection of the right resonant boundary with
    the subcritical bound. Use points along the bound to give as
    guess to fsolve. When fsolve soln comes close to the guess,
    return the current guess.

    F((h, d11), u0) is a function for fsolve to use to evaluate
    the rightward bound.

    d0 is the interface depth

    d is half the width of the window that the guess must fall within

    bound is a tuple (U, H) where U and H are arrays of the velocity
    and current depth along the subcritical boundary.
    """
    if not bound:
        lower_branch, upper_branch = critical_bounds(d0)
    elif bound:
        lower_branch = bound
    # sort by u
    p = lower_branch[0].argsort()
    bound_U = lower_branch[0][p]
    bound_h = lower_branch[1][p]
    for i, u in enumerate(bound_U):
        h = bound_h[i]
        # use d0 as guess for d11
        _h, _d11 = fsolve(F, (h, d0), args=(u,))
        if h - d < _h < h + d:
            return h, _d11, u
    return False


def right_resonance(S_=0.75, d0_=0.3, lower_bound=None, res=100):
    """The method used by White and Helfrich to find the
    right bound of the resonant wedge is as follows.

    Equations 3.1, 3.3, 3.4, 3.4 and 3.6 are valid at all
    points inside the resonant wedge. Equation 3.8 is valid
    at all points if the dissipation is left to vary. The
    dissipation is zero along the rightward bound.

    Equation 3.9 is used as a criterion to determine when
    resonant solutions can no longer exist, i.e. for each
    solution point we find we check whether 3.9 is valid.

    3.3 and 3.4 are used to subsitute out U11 and U21,
    leaving 4 equations in 7 unknowns.

    We specify d0 and S and vary U0 incrementally,
    leaving 4 equations in 4 unknowns, (h, d11, d1c, Cb).

    We can eliminate Cb using 3.1 and any of (h, d11, d1c)
    using 3.8 (given a value for the dissipation, zero
    along the right bound).

    This reduces the system to 2 equations in two unknowns
    for a given value of U0.

    Given an initial guess we can use fsolve to solve
    this system.

    We do this for a range of U0 from the subcritical bound
    up to the upper limit U0=0.5. To find the first point at
    the lower limit we can use pairs of (U0, h) known to lie
    along the subcritical bound as the initial guess and
    begin to deviate from the bound when the guess is close to
    the solution.

    Along the right branch, we use the previous solution
    as the next guess whilst incrementing U0.

    The upper limit of the right branch is found when the
    long wave speed through the bore, given by 3.9, changes
    sign.

    Thus we evaluate 3.9 for each point found on the right
    branch and terminate the branch when this approaches
    zero.

    Inputs: values for d0 and S.

    Returns: an array of points on the branch. For each point,
             all of the variables are evaluated.
    """
    # u11, u21 as f(U0, d0, d11)
    u11, u21 = us()
    # solve eq38 for d1c (positive root)
    # with dissipation equal to zero
    # and sub out u11 to give d1c = f(U0, d0, S, h, d11)
    d1c_ = sp.solve(eq38(Dc=0), d1c)[1].subs({U11: u11})
    # now sub out S, d0, d1c, u11, u21 from eqs 3.5 and 3.6
    # have to do S and d0 in additional step to get them all out
    subs = {d1c: d1c_, U11: u11, U21: u21}
    s35 = eq35().subs(subs).subs({S: S_, d0: d0_})
    s36 = eq36().subs(subs).subs({S: S_, d0: d0_})
    # s35 and s36 are two equations in (h, d11, U0)
    f35 = sp.lambdify((h, d11, U0), s35, "numpy")
    f36 = sp.lambdify((h, d11, U0), s36, "numpy")

    # function for fsolve
    def E(p, U0):
        """p is a tuple (h, d11)"""
        return f35(*p, U0=U0), f36(*p, U0=U0)

    # find the initial guess using the subcritical boundary
    _h, _d11, _U0 = f_subcrit(E, d0=d0_, d=0.005, bound=lower_bound)
    print("Starting branch at ({u}, {h})".format(u=_U0, h=_h))
    guess = _h, _d11
    p = guess
    branch = []
    U = np.linspace(_U0, 0.5, res)
    for u0 in U:
        p = fsolve(E, p, args=(u0))
        branch.append((p[0], p[1], u0))

    branch = np.asarray(branch)
    # H = branch[:, 0]
    D11 = branch[:, 1]
    U = branch[:, 2]

    # evaluate critical criterion
    crit = eq39(U11=u11, U21=u21)
    fcrit = sp.lambdify((U0, d0, d11), crit, "numpy")
    fc = fcrit(U, d0_, D11)
    zeros = np.where(np.diff(np.sign(fc)))[0]
    if len(zeros) == 0:
        return branch
    else:
        # first zero crossing
        zero = zeros[0]
        # return the branch up to the first zero crossing of criterion
        return branch[:zero]

    # evaluations
    # u11, u21, cb, d1c
    # fu11 = sp.lambdify((U0, d0, d11), u11, "numpy")
    # fu21 = sp.lambdify((U0, d0, d11), u21, "numpy")
    # fcb = sp.lamdify((U0, d0, d11), sp.solve(eq31(), Cb), "numpy")
    # fd1c = sp.lambdify((U0, d0, d11, S, h), d1c_, "numpy")

### /RIGHT BOUND ###


### AMPLITUDES ###
def d11_contours():
    """The contours in d11 are computed by varying the
    dissipation in eq 3.8.

    Zero dissipation defines the right edge of the
    resonant wedge. Inside the wedge the dissipation
    varies.

    The upper bound of the resonant wedge is given by
    a d11 contour.
    """
    pass

### /AMPLITUDES ###


### DO STUFF ###
def make_regime_diagram(S_=0.75, d0_=0.3, res=100):
    # H = np.linspace(0, d0, res)
    lower_critical, upper_critical = critical_bounds(d0=d0_)
    branch = right_resonance(S_=S_, d0_=d0_, lower_bound=lower_critical)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # lower critical bound
    ax.plot(lower_critical[1], lower_critical[0])
    # upper critical bound
    ax.plot(upper_critical[1], upper_critical[0])
    # line at 0.5 originating from furthest right point of upper
    # bound
    h = np.linspace(upper_critical[1].max(), 1)
    u = np.array([0.5 for i in h])
    ax.plot(h, u)
    # right resonant bound
    ax.plot(branch[:, 0], branch[:, 2])
    plt.show()
    return upper_critical

### /DO STUFF ###


def brentq_scan(f, a, b, d=None, n=1):
    """Uses the brentq optimization routine to look for multiple
    roots on the interval [a, b], by dividing it into either smaller
    intervals of width d or by dividing it into a given number of
    subsections.

    Returns a list of roots
    """
    if d:
        n = int((b - a) / d)
    if not d:
        d = int((b - a) / n)
    ints = np.linspace(a, b, n)
    sints = ints + d
    roots = []
    for a1, b1 in zip(ints, sints):
        try:
            roots.append(brentq(f, a1, b1))
        # catch no sign change on interval
        except ValueError:
            pass
    return roots


def Uh(d0, U_sol, h_sol, D0):
    # The U and h arrays are two dimensional and indexed with d0 as
    # the first axis and d1c as the second.
    # To get U and h for a specific d0 (over all d1c)
    # we need to find the nearest index to the d0 that we want
    idx = np.abs(D0 - d0).argmin()
    U = [u[idx] for u in U_sol]
    H = [h[idx] for h in h_sol]

    return U, H


def plot_branch(ax, n, U, H):
    u = U[n]
    h = H[n]
    ax.plot(h, u)
    return ax


def main(d0_=0.3):
    H = np.linspace(0.001, 0.999, 1000)
    poly = U_subbed_poly()
    Usol = fast_solve(H, d0_=d0_, combined_poly=poly)
    print "extracting U and h at d0 = %s" % d0_
    # U, H = Uh(0.1, Usol, hsol, D0)
    print "plotting first solution branch..."
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for n in range(4):
        # plot_branch(n, U, H, ax)
        ax.plot(H, Usol[n], 'o')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.savefig('branches.png')


if __name__ == '__main__':
    main()
