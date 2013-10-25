import sympy as sp
import numpy as np

from scipy.optimize import fsolve

# wave amplitudes, speed
a, b, c = sp.symbols('a, b, c')

# stratification parameter
s = sp.symbols('s')

# unperturbed layer speeds
u1, u2, u3 = sp.symbols('u1, u2, u3')

# scaled layer speeds vi = 1 - ui / c
v1, v2, v3 = sp.symbols('v1, v2, v3')

# unperturbed layer depths
h1, h2, h3 = sp.symbols('h1, h2, h3')


def find_intersections(A, B):
    # min, max and all for arrays
    amin = lambda x1, x2: np.where(x1<x2, x1, x2)
    amax = lambda x1, x2: np.where(x1>x2, x1, x2)
    aall = lambda abools: np.dstack(abools).all(axis=2)
    slope = lambda line: (lambda d: d[:,1]/d[:,0])(np.diff(line, axis=0))

    x11, x21 = np.meshgrid(A[:-1, 0], B[:-1, 0])
    x12, x22 = np.meshgrid(A[1:, 0], B[1:, 0])
    y11, y21 = np.meshgrid(A[:-1, 1], B[:-1, 1])
    y12, y22 = np.meshgrid(A[1:, 1], B[1:, 1])

    m1, m2 = np.meshgrid(slope(A), slope(B))
    m1inv, m2inv = 1/m1, 1/m2

    yi = (m1*(x21-x11-m2inv*y21) + y11)/(1 - m1*m2inv)
    xi = (yi - y21)*m2inv + x21

    xconds = (amin(x11, x12) < xi, xi <= amax(x11, x12),
              amin(x21, x22) < xi, xi <= amax(x21, x22) )
    yconds = (amin(y11, y12) < yi, yi <= amax(y11, y12),
              amin(y21, y22) < yi, yi <= amax(y21, y22) )

    return xi[aall(xconds)], yi[aall(yconds)]

def F(a=a, b=b, v1=v1, v2=v2, v3=v3, h1=h1, h2=h2, h3=h3, s=s):
    A = (h1 + a) ** 2
    B = (h2 + b - a) ** 2
    C = (h3 - b) ** 2

    f1 = - (a + b * s) * v2 ** 2 * h2 ** 2 * A * C
    f2 = B * (a * v3 ** 2 * h3 ** 2 * A + b * s * v1 ** 2 * h1 ** 2 * C)
    f3 = A * B * C * (a * (v2 ** 2 - v3 ** 2) + b * s * (v2 ** 2 - v1 ** 2))

    return f1 + f2 + f3


def G(a=a, b=b, v1=v1, v2=v2, v3=v3, h1=h1, h2=h2, h3=h3):
    A = (h1 + a) ** 2
    B = (h2 + b - a) ** 2
    C = (h3 - b) ** 2

    fu1 = v1 ** 2 * a ** 3 / A
    fu2 = v2 ** 2 * (b - a) ** 3 / B
    fu3 = v3 ** 2 * b ** 3 / C

    return fu1 + fu2 - fu3


class FGSolver(object):
    """For given vi, hi, s, find the possible solutions in (a, b)."""
    def __init__(self, s=1, h1=0.2, h2=0.6, h3=0.2, v1=1, v2=1, v3=1):
        # Create functions of (a, b) for these given parameters
        self.F = F(v1=v1, v2=v2, v3=v3, h1=h1, h2=h2, h3=h3, s=s)
        self.G = G(v1=v1, v2=v2, v3=v3, h1=h1, h2=h2, h3=h3)
        self.f = sp.lambdify((a, b), self.F)
        self.g = sp.lambdify((a, b), self.G)

    def rough_zero(self, start=0.0001, end=1, res=1000):
        """roughly calculate zero contours of F.

        Just does mode-1 for now (a and b same sign).

        Returns two arrays of shape (2, res), corresponding to the
        zero contour in the upper right and lower left quadrants.
        """
        # upper right quadrant

        B = np.linspace(start, end, res)
        a1 = [start]
        for b in B:
            # use last value of a as guess
            a = fsolve(self.f, a1[-1], args=(b,))
            a1.append(a)
        a1 = np.hstack(a1[1:])

        s1 = np.vstack((a1, B))

        B = -np.linspace(start, end, res)
        a2 = [-start]
        for b in B:
            # use last value of a as guess
            a = fsolve(self.f, a2[-1], args=(b,))
            a2.append(a)
        a2 = np.hstack(a2[1:])

        s2 = np.vstack((a2, B))

        return s1, s2

    def global_rough_zero(self, f):
        x = np.linspace(-1, 1, 1000)
        y = np.linspace(-1, 1, 1000)
        A, B = np.meshgrid(x, y)
        fab = f(A, B)

        # find rough zeros in F over the domain
        zeros = np.where(np.diff(np.sign(fab)) != 0)
        a, b = A[zeros], B[zeros]

        # TODO:just have to find a way of determing where G changes sign
        # on these points
        return a, b

    def zeroG(self, f0):
        """Evaluate G along the lines F=0.

        We are looking G=0, i.e. a sign change.

        Inputs:
            f0 - an array of (a, b), shape (2, N)

        Return (a, b) in the vicinity of the sign change.
        """
        g = self.g(*f0)
        # find zero crossing
        # TODO: more elegant? like interpolating?
        zero = np.where(np.diff(np.sign(g)) != 0)
        ab = np.vstack((f0[0, zero], f0[1, zero])).squeeze()
        return ab

    def enhance(self, guess, **kwargs):
        """Using a rough guess for (a, b), converge on the
        zero using a non linear solver with a newton raphson
        method.
        """
        # nsolve can't take arrays as input for some reason
        guess = tuple(guess)
        variables = (a, b)
        equation_set = (self.F, self.G)
        ab = sp.nsolve(equation_set, variables, guess, **kwargs)
        return np.array(ab, dtype=float)

    @property
    def roots(self):
        """Calculate the roots of the system."""
        s1, s2 = self.rough_zero()
        guess1 = self.zeroG(s1)
        guess2 = self.zeroG(s2)

        ab1 = self.enhance(guess1)
        ab2 = self.enhance(guess2)

        return ab1, ab2


# evaluate c+- over a, b

# this doesn't seem well conditioned?

# alternative: formulate lambs base equations as 3 equations in
# (a, b, c).
# pass these directly to either fsolve or nsolve as a system of
# equations.

# or use scipy.optimize.root - compute jacobian analytically?

# are these going to pick up a=b=c=0 as trivial solution??
# how many roots are there going to be? at least two, probably 4.

# The equations in Lamb's notation:

class LambBase(object):
    @staticmethod
    def f1(a=a, b=b, c=c, s=s, u1=u1, u2=u2, u3=u3, h1=h1, h2=h2, h3=h3):
        A = (h1 + a)
        B = (h2 + b - a)
        C = (h3 - b)

        fu1 = (c - u1) ** 2 * (1 - (h1 / A) ** 2)
        fu2 = (c - u2) ** 2 * (1 - (h2 / B) ** 2)

        return fu1 - fu2 - 2 * a / s

    @staticmethod
    def f2(a=a, b=b, c=c, s=s, u1=u1, u2=u2, u3=u3, h1=h1, h2=h2, h3=h3):
        A = (h1 + a)
        B = (h2 + b - a)
        C = (h3 - b)

        fu2 = (c - u2) ** 2 * (1 - (h2 / B) ** 2)
        fu3 = (c - u3) ** 2 * (1 - (h3 / C) ** 2)

        return fu2 - fu3 - 2 * b

    @staticmethod
    def f3(a=a, b=b, c=c, s=s, u1=u1, u2=u2, u3=u3, h1=h1, h2=h2, h3=h3):
        A = (h1 + a)
        B = (h2 + b - a)
        C = (h3 - b)

        fu1 = (c - u1) ** 2 * a ** 3 / A ** 2
        fu2 = (c - u2) ** 2 * (b - a) ** 3 / B ** 2
        fu3 = (c - u3) ** 2 * b ** 3 / C ** 2

        return fu1 + fu2 - fu3

    A = (h1 + a)
    B = (h2 + b - a)
    C = (h3 - b)

    H1 = A - h1 ** 2
    H2 = A - h2 ** 2
    H3 = A - h3 ** 2

    alpha = 2 * a / s
    beta = 2 * b

    def cu1(self):
        """(c - U1) ** 2 as a function of (a, b)."""
        n = H3 * alpha * (b - a) ** 3 - H2 * (alpha + beta) * b ** 3
        d = H2 * H3 * a ** 3 + H1 * H3 * (b - a) ** 3 - H1 * H2 * b ** 3
        return A ** 2 * n / d

    def cu2():
        """(c - U2) ** 2 as a function of (a, b), given (c - U1) ** 2."""
        return (cu1() / A ** 2 * H1 - alpha) / H2

    def cu3():
        """(c - U3) ** 2 as a function of (a, b), given (c - U1) ** 2."""
        return (cu1() / A ** 2 * H1 - (alpha + beta)) / H3




class Example_nsolve(object):
    def __init__(self, s=s, u1=u1, u2=u2, u3=u3, h1=h1, h2=h2, h3=h3):
        """Solve the base equations for (a, b, c) using sympy's
        nsolve. All the arguments need to be specified."""
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3

        self.eq1 = LambBase.f1(s=s, u1=u1, u2=u2, u3=u3, h1=h1, h2=h2, h3=h3)
        self.eq2 = LambBase.f2(s=s, u1=u1, u2=u2, u3=u3, h1=h1, h2=h2, h3=h3)
        self.eq3 = LambBase.f3(s=s, u1=u1, u2=u2, u3=u3, h1=h1, h2=h2, h3=h3)

        self.equation_set = (self.eq1, self.eq2, self.eq3)
        self.variables = (a, b, c)

    def solve(self, guess, **kwargs):
        try:
            abc = sp.nsolve(self.equation_set, self.variables, guess, **kwargs)
        except ValueError:
            # non convergence of solver
            abc = None
        return abc

    def array_of_guesses(self, res=10):
        # TODO: it would be faster to search the space for the known
        # number of solutions with bisection

        # array within the solution triangle
        h1, h2, h3 = self.h1, self.h2, self.h3
        # array over a and b without including edges
        a = np.linspace(-h1 + h1 / res, 1 - h1, res, endpoint=False)
        b = np.linspace((-h1 - h2) + h1 / res, h3, res, endpoint=False)

        c = np.linspace(-1, 1, res)

        A, B, C = np.meshgrid(a, b, c)

        # final constraint on the triangle
        valid = np.where((-h2 < (B - A)) & ((B - A) < (1 - h2)))

        vA, vB, vC = A[valid], B[valid], C[valid]

        return zip(vA, vB, vC)

    def all_solutions(self):
        guesses = self.array_of_guesses()
        return guesses


class SolveGivenU(object):
    pass






# TODO: create an array of initial guesses in the space (a, b, c)
# bounded by the solution triangle.

# TODO: catch non convergence of solver
# TODO: condense array of solutions to unique solutions.
# TODO: divide these into mode-1, mode-2

# TODO: for a specific (S, d0) compute the supercritical solution
# curve, defining (h, U0, d1c) along its length.
# For each point on the curve, compute conjugate solutions (a, b, c)
#
# Is there a point on the curve at which U0 = c? i.e. the gravity
# current is going at the same speed as the conjugate wave speed?
#
# Does c vary monotonically along the curve?
#
# If there is a point U0=c, does it match the transition from type
# IV to type V behaviour?
