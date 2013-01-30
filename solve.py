import numpy as np
# import matplotlib as mpl
# mpl.use('Agg')
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import brentq, fsolve
from sympy.utilities.lambdify import lambdify

h, U0, d1c, d0 = sp.symbols('h, U0, d1c, d0')
Cb, U11, U21, d11, S = sp.symbols('Cb, U11, U21, d11, S')


def sympy_solve():
    h, U0, d1c = sp.S('h U0 d1c'.split())
    d0 = 0.1
    equations = [(U0) ** 2 * ((d0 ** 2 / d1c ** 3) + (1 - d0) ** 2 / (1 - d1c - d0 * h) ** 3) - 1,
                 0.5 * (U0) ** 2 * ((d0 ** 2 / d1c ** 2) + (1 - d0) ** 2 / (1 - d1c - d0 * h)) + d1c + d0 * (h - 1)]
    sp.nsolve(equations, (U0, h, d1c), (0.5, 0.5, 0.2))


class Equation():
    """A factory for the equations found in the paper"""
    def __init__(self, eqno):
        self.eqno = eqno

    h, U0, d1c, d0 = sp.symbols('h, U0, d1c, d0')

    def eq211():
        pass


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
            + (U11 ** 2) * (0.5 + (d11 ** 2 / d1c) - d11)\
            + (U21 ** 2) * ((1 - d11) ** 2 / (1 - d1c - h) + d11 - 1)
    return f


def eq38(U0=U0, S=S, h=h, d11=d11, d1c=d1c, U11=U11):
    """Energy dissipation, with Dc = 0. Rightward bound of resonant wedge."""
    f = h * ((1 - S) / S) - (U11 ** 2 / 2) * (d11 ** 2 / d1c ** 2)
    return f


def eq39(U0=U0, d11=d11, U11=U11, U21=U21):
    """Bore criticality. Upper bound of resonant wedge."""
    # TODO: verify this
    f = U0 + (U11 - U21) * (1 - 2 * d11) - ((1 - (U11 - U21) ** 2) * d11 * (1 - d11)) ** .5
    return f


def us():
    """Substitute out the bore speed to obtain relations for u11 and u12.
    """
    # TODO: check sign of roots
    # I think this is the correct root for Cb [1]
    # actually, taking the first one seems to get the right
    # answers...
    cb = sp.solve(eq31(), Cb)[0]
    u11 = sp.solve(eq33().subs(Cb, cb), U11)[0]
    u21 = sp.solve(eq34().subs(Cb, cb), U21)[0]
    return u11, u21


def resonance():
    """Reduce the set of resonant equations to two equations in
    (S, d0, d11, U0, h, d1c)

    This uses equations 3.1, 3.3, 3.4 (through us()), 3.5 and 3.6
    so should represent two equations that hold over the whole
    resonant region.
    """
    u11, u21 = us()
    # now use these in eq35 and eq36 to elim u11, u21, d1c
    p35 = eq35().subs({U11: u11, U21: u21})
    p36 = eq36().subs({U11: u11, U21: u21})
    # should be two equations in 6 unknowns (S, d0, d11, U0, h, d1c)

    # u21b = sp.solve(eq35(), U21)[0]
    # right_bound = eq36().subs({U21: u21b, U21: u11, d1c: d1c_})
    return p35, p36


def bore_amp_contour(d11_, h_, guess):
    """Find a specific contour of bore amplitude d11_ for
    given h_, starting with the guess in (U0, d1c).
    Returns the converged U0, d1c.
    """
    S_ = 0.75
    d0_ = 0.3
    p35, p36 = resonance()
    p35d, p36d = p35.subs({d11: d11_}), p36.subs({d11: d11_})
    p35s, p36s = p35d.subs({S: S_, d0: d0_}), p36d.subs({S: S_, d0: d0_})
    p35sh, p36sh = p35s.subs(h, h_), p36s.subs(h, h_)

    f35 = sp.lambdify((U0, d1c), p35sh, "numpy")
    f36 = sp.lambdify((U0, d1c), p36sh, "numpy")

    def E(p):
        return f35(*p), f36(*p)
    root = fsolve(E, guess)
    return root


def right_resonance(U, d11_=None, S_=0.75, d0_=0.3, guess=(0.2, 0.2)):
    """For a given h, S, d0 and optionally d11, find the U0 that
    corresponds to the rightward bound of the resonant wedge.

    TODO: change this to specifying U0 and finding h from a guess in (h, d11)


    Requires an initial guess for U0, d11 unless d11 is specified in
    which case the guess is for U0 only.
    """
    u11 = us()[0]
    d1c_ = sp.solve(eq38(), d1c)[1].subs(U11, u11)
    p35, p36 = resonance()
    p35d, p36d = p35.subs({d1c: d1c_}), p36.subs({d1c: d1c_})
    p35s, p36s = p35d.subs({S: S_, d0: d0_}), p36d.subs({S: S_, d0: d0_})

    root = []
    for u_ in U:
        p35su, p36su = p35s.subs(U0, u_), p36s.subs(U0, u_)

        if not d11_:
            f35 = sp.lambdify((h, d11), p35su, "numpy")
            f36 = sp.lambdify((h, d11), p36su, "numpy")
        else:
            p35sh, p36sh = p35sh.subs(d11, d11_), p36sh.subs(d11, d11_)
            f35 = sp.lambdify((U0), p35sh, "numpy")
            f36 = sp.lambdify((U0), p36sh, "numpy")

        def E(p):
            return f35(p[0], p[1]), f36(p[0], p[1])
        # return E
        root.append(fsolve(E, guess))
    return root


def upper_resonance(d0_=0.3, S_=0.75):
    u11, u21 = us()
    eq39s = eq39().subs({U11: u11, U21: u21})

    D11 = sp.solve(eq39s, d11)
    # this should be d11(U0, d0), but there will be two solns.
    # FIXME: you can't do this. it isn't analytically soluble in
    # d11.

    # now we obtain two equations in (U0, h, S, d1c, d0)
    # into which we sub in S, d0 to get two in (U0, h, d1c)
    eq35d = eq35().subs({d11: D11[0], S: S_, d0: d0_})
    eq36d = eq35().subs({d11: D11[0], S: S_, d0: d0_})
    # for given h, these can be solved numerically
    f35 = sp.lambdify((U0, d11, h), eq35d, "numpy")
    f36 = sp.lambdify((U0, d11, h), eq35d, "numpy")
    def E(p, h):
        return f35(p[0], p[1], h), f36(p[0], p[1], h)
    return E
    root = fsolve(E, guess)



def subbed():
    f1 = eq211()
    f2 = eq212()
    # rearrange f1 to get U0^2
    U2 = ((f1 + 1) / U0 ** 2) ** -1
    # substitute this into f2
    f3 = f2.subs(U0 ** 2, U2)
    return f3


def simplify():
    exp = subbed()
    simp = sp.simplify(exp)
    # simp = sp.collect(exp, d1c)
    print simp


def so_sol(d0v=0.1):
    h, U0, d1c, d0 = sp.symbols('h, U0, d1c, d0')
    f1 = eq211(h, U0, d1c, d0)
    f2 = eq212(h, U0, d1c, d0)

    # rearrange f1 to get U0^2
    U2 = ((f1 + 1) / U0 ** 2) ** -1
    # substitute this into f2
    f3 = f2.subs(U0 ** 2, U2)

    # f3 is now f(h, d1c) only.
    # integer exponents --> rational fraction
    # so we only need to take the numerator of f3 as we are
    # looking for where it goes to 0. this is a polynomial in
    # (h, d1c, d0)
    print "constructing polynomial in h..."
    p3 = sp.fraction(f3.cancel())[0]

    # subsitute in a value of d0 (don't actually have to, can leave
    # this unconstrained if we want).
    hd1c = p3.subs(d0, d0v)
    U2 = U2.subs(d0, d0v)
    # hd1c = p3

    # hd1c is a quartic in h and a quintic in d1c.  there is no
    # algorithm to solve a quintic, but quartics are soluble.
    print "solving for h..."
    hsol = sp.solve(hd1c, h)
    return hsol, U2


def brent_sol():
    f1 = eq211()
    f2 = eq212()

    h, U0, d1c, d0 = sp.symbols('h, U0, d1c, d0')
    # rearrange f1 to get U0^2
    U2 = ((f1 + 1) / U0 ** 2) ** -1
    # substitute this into f2
    f3 = f2.subs(U0 ** 2, U2)

    # f3 is now f(h, d1c) only.
    # integer exponents --> rational fraction
    # so we only need to take the numerator of f3 as we are
    # looking for where it goes to 0. this is a polynomial in
    # (h, d1c, d0)
    print "constructing polynomial in h..."
    p3 = sp.fraction(f3.cancel())[0]

    # subsitute in a value of d0 (don't actually have to, can leave
    # this unconstrained if we want).
    d0v = 0.1
    hd1c = p3.subs(d0, d0v)

    # hd1c is a quartic in h and a quintic in d1c.  there is no
    # algorithm to solve a quintic, but quartics are soluble.

    # let's solve the quintic numerically over a range of h
    H = np.linspace(0.1, 1, 10)

    quintic = lambdify((d1c, h), hd1c, "numpy")
    return quintic

    c0 = (d0v * (1 - d0v)) ** .5
    print "applying brentq"
    # the subcritical boundary is on the interval [0, c0]
    # yes, but this boundary is in U and the quintic is in d1c!
    U_sub = np.array([brentq(quintic, 0, c0, args=(h)) for h in H])
    # supercritical boundary is on [c0, 0.5]
    U_super = np.array([brentq(quintic, c0, 0.5, args=(h)) for h in H])

    return U_sub, U_super


def U(hv, d0v, d1cv):
    f1 = eq211()
    # rearrange f1 to get U0^2
    U2 = ((f1 + 1) / U0 ** 2) ** -1
    U = U2.subs({h: hv, d0: d0v, d1c: d1cv}) ** .5
    return U


def U_subbed_poly():
    f1 = eq211()
    f2 = eq212()
    # rearrange f1 to get U0^2
    U2 = ((f1 + 1) / U0 ** 2) ** -1
    # substitute this into f2
    f3 = f2.subs(U0 ** 2, U2)
    # f3 is now f(h, d1c) only.
    # integer exponents --> rational fraction
    # so we only need to take the numerator of f3 as we are
    # looking for where it goes to 0. this is a polynomial in
    # (h, d1c, d0)
    print "constructing polynomial in d1c..."
    p3 = sp.fraction(f3.cancel())[0]
    return p3


def fast_solve(H, d0_=0.1, combined_poly=None):
    if not combined_poly:
        combined = U_subbed_poly()
    combined = combined_poly
    # for h_ in np.linspace(0, 1):
        # p = sp.Poly(combined.subs({h: h_, d0: d0_}), d1c)
        # D1c = np.roots(p.coeffs())
        # u = [U(h_, d0_, d1c_) for d1c_ in D1c
                    # if np.isreal(d1c_) and (0 < d1c_.real < h_)]
    # TODO: lambdify, work over array of h
    # we want to get the coefficients, but also to be able to
    # evaluate over an input array. the latter requires that we
    # use lambdify.
    # we can do this!
    coeff = lambdify((d0, h), sp.Poly(combined, d1c).coeffs(), "numpy")
    d0_ = np.array([d0_])
    coeffs = np.array(coeff(*np.meshgrid(d0_, H)))
    # now we have a list of arrays of coefficients, each array
    # having the dimension of H
    # np.roots can only take 1d input so we have to be explicit
    Roots = np.array([np.roots(coeffs[:, i].squeeze()) for i in range(len(H))])
    # Roots is now a 2d array of first dimension H. That is, for h
    # in H, roots in d1c are given by roots in Roots.
    # Now we work out what U would be
    Uf = lambdify((d0, h, d1c), U(h, d0, d1c), "numpy")
    U_sol = [Uf(d0_, H, Roots[:, i]) for i in range(4)]
    return U_sol


    # what we want is an array of h and 3 arrays of corresponding
    # roots in U. then if we also have an input array of d0, we
    # can have 2d arrays in U.


def poly_solve(p3, hv, d0v):
    """for given d0, h find roots of the polynomial p3"""
    phd = p3.subs({h: hv, d0: d0v})
    coeffs = sp.poly(phd, d1c).coeffs()
    # TODO: substitute the coeffs just before calculating the roots
    roots = np.roots(coeffs)
    # and throw out those that aren't physical.
    physical_roots = [r for r in roots if (0 < r < (1 - hv))]
    # then calculate U0 from the given d0, h for each of the roots
    # in d1c
    return roots
    return physical_roots


def h_eval(hsol, U2, n=1000):
    h, U0, d1c, d0 = sp.symbols('h, U0, d1c, d0')
    # hsol is 4 roots of a quartic. for each of these roots we need to
    # evaluate h over a fine range of physical d1c to give an array
    # of possible (h, d1c) for each root.

    # evaluation over an array is slow using sp. we use lambdify
    # to convert to a function that can be used for fast evaluation.
    # solF is a list of functions of d1c and d0 representing the roots
    # solF = [lambdify((d1c, d0), sol) for sol in hsol]
    solF = [lambdify((d1c), sol, "numpy") for sol in hsol]

    # D0 = np.linspace(0.001, 1, 1000)
    D1c = np.linspace(0.001 + 0j, 1, n)
    # grid = np.meshgrid(D1c, D0)
    print "evaluating h over grid"
    # h_roots_eval = [f(*grid) for f in solF]
    h_roots_eval = [f(D1c) for f in solF]

    # we then subsitute these arrays of h, d1c, d0 back into f1 or f2
    # to find the corresponding U0, giving us an array of (U0, h,
    # d1c) for each root.

    U = lambdify((h, d1c), U2 ** .5, "numpy")
    print "evaluating U over grid"
    Usol = [U(h, D1c) for h in h_roots_eval]

    # Useful outputs are
    # D0 - array of d0
    # D1c - array of d1c
    # h_roots_eval - list of arrays of h evaluated over a grid made
    #                from D0 and D1c, one element per root of the h
    #                quartic.
    # Usol - U0 that correspond to the h in h_roots_eval

    return Usol, h_roots_eval
    # return Usol, h_roots_eval, D0


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


def main():
    H = np.linspace(0.001, 0.999, 1000)
    poly = U_subbed_poly()
    Usol = fast_solve(H, d0_=0.1, combined_poly=poly)
    print "extracting U and h at d0 = 0.1"
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
