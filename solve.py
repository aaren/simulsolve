import numpy as np
import sympy
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from sympy.utilities.lambdify import lambdify


def sympy_solve():
    h, U0, d1c = sympy.S('h U0 d1c'.split())
    d0 = 0.1
    equations = [(U0) ** 2 * ((d0 ** 2 / d1c ** 3) + (1 - d0) ** 2 / (1 - d1c - d0 * h) ** 3) - 1,
                 0.5 * (U0) ** 2 * ((d0 ** 2 / d1c ** 2) + (1 - d0) ** 2 / (1 - d1c - d0 * h)) + d1c + d0 * (h - 1)]
    sympy.nsolve(equations, (U0, h, d1c), (0.5, 0.5, 0.2))


def eq1(h, U0, d1c, d0):
    f = (U0) ** 2 * ((d0 ** 2 / d1c ** 3) + (1 - d0) ** 2 / (1 - d1c - d0 * h) ** 3) - 1
    return f


def eq2(h, U0, d1c, d0):
    f = 0.5 * (U0) ** 2 * ((d0 ** 2 / d1c ** 2) + (1 - d0) ** 2 / (1 - d1c - d0 * h)) + d1c + d0 * (h - 1)
    return f


def subbed():
    h, U0, d1c, d0 = sympy.S('h U0 d1c d0'.split())
    exp = 0.5 * (1 / ((d0 ** 2 / d1c ** 3) + (1 - d0) ** 2 / (1 - d1c - d0 * h) ** 3)) * ((d0 ** 2 / d1c ** 2) + (1 - d0) ** 2 / (1 - d1c - d0 * h)) + d1c + d0 * (h - 1)
    return exp


def simplify():
    exp = subbed()
    simp = sympy.simplify(exp)
    # simp = sympy.collect(exp, d1c)
    print simp


def so_sol():
    h, U0, d1c, d0 = sympy.symbols('h, U0, d1c, d0')
    f1 = eq1(h, U0, d1c, d0)
    f2 = eq2(h, U0, d1c, d0)

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
    p3 = sympy.fraction(f3.cancel())[0]

    # subsitute in a value of d0 (don't actually have to, can leave
    # this unconstrained if we want).
    # d0v = 0.1
    # hd1c = p3.subs(d0, d0v)
    hd1c = p3

    # hd1c is a quartic in h and a quintic in d1c.  there is no
    # algorithm to solve a quintic, but quartics are soluble.
    print "solving for h..."
    hsol = sympy.solve(hd1c, h)
    return hsol, U2


def brent_sol():
    h, U0, d1c, d0 = sympy.symbols('h, U0, d1c, d0')
    f1 = eq1(h, U0, d1c, d0)
    f2 = eq2(h, U0, d1c, d0)

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
    p3 = sympy.fraction(f3.cancel())[0]

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


def h_eval(hsol, U2):
    h, U0, d1c, d0 = sympy.symbols('h, U0, d1c, d0')
    # hsol is 4 roots of a quartic. for each of these roots we need to
    # evaluate h over a fine range of physical d1c to give an array
    # of possible (h, d1c) for each root.

    # evaluation over an array is slow using sympy. we use lambdify
    # to convert to a function that can be used for fast evaluation.
    # solF is a list of functions of d1c and d0 representing the roots
    solF = [lambdify((d1c, d0), sol) for sol in hsol]

    D0 = np.arange(0, 1, 11)
    D1c = np.arange(0, 1, 11)
    grid = np.meshgrid(D1c, D0)
    print "evaluating h over grid"
    h_roots_eval = [f(*grid) for f in solF]

    # we then subsitute these arrays of h, d1c, d0 back into f1 or f2
    # to find the corresponding U0, giving us an array of (U0, h,
    # d1c) for each root.

    U = lambdify((h, d1c, d0), U2 ** .5)
    print "evaluating U over grid"
    Usol = [U(h, *grid) for h in h_roots_eval]

    # Useful outputs are
    # D0 - array of d0
    # D1c - array of d1c
    # h_roots_eval - list of arrays of h evaluated over a grid made
    #                from D0 and D1c, one element per root of the h
    #                quartic.
    # Usol - U0 that correspond to the h in h_roots_eval

    return Usol, h_roots_eval, D0


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
    hsol, U2 = so_sol()
    Usol, hsol, D0 = h_eval(hsol, U2)
    print "extracting U and h at d0 = 0.3"
    U, H = Uh(0.3, Usol, hsol, D0)
    print "plotting first solution branch..."
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for n in range(4):
        plot_branch(n, U, H, ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.savefig('branches.png')


if __name__ == '__main__':
    main()
