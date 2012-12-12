# import numpy as np
import sympy
# from scipy.optimize import fsolve


def sympy_solve():
    h, U0, d1c = sympy.S('h U0 d1c'.split())
    d0 = 0.1
    equations = [(U0) ** 2 * ((d0 ** 2 / d1c ** 3) + (1 - d0) ** 2 / (1 - d1c - d0 * h) ** 3) - 1,
                 0.5 * (U0) ** 2 * ((d0 ** 2 / d1c ** 2) + (1 - d0) ** 2 / (1 - d1c - d0 * h)) + d1c + d0 * (h - 1)]
    sympy.nsolve(equations, (U0, h, d1c), (0.5, 0.5, 0.2))


def eq1(h, U0, d1c, d0=0.1):
    f = (U0) ** 2 * ((d0 ** 2 / d1c ** 3) + (1 - d0) ** 2 / (1 - d1c - d0 * h) ** 3) - 1
    return f


def eq2(h, U0, d1c, d0=0.1):
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
