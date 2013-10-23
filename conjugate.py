import sympy as sp

# wave amplitudes, speed
a, b, c = sp.symbols('a, b, c')

# stratification parameter
s = sp.symbols('s')

# unperturbed layer speeds
u1, u2, u3 = sp.symbols('u1, u2, u3')

# unperturbed layer depths
h1, h2, h3 = sp.symbols('h1, h2, h3')


def F(a=a, b=b, c=c, s=s, u1=u1, u2=u2, u3=u3, h1=h1, h2=h2, h3=h3):
    A = (h1 + a) ** 2
    B = (h2 + b - a) ** 2
    C = (h3 - b) ** 2

    fu1 = (c - u1) ** 2 * b * s * (a ** 2 + 2 * a * h1) * A * B
    fu2 = (c - u2) ** 2 * (a + b * s) * ((b - a) ** 2
                                         + 2 * (b - a) * h2) * A * C
    fu3 = (c - u3) ** 2 * a * (b ** 2 - 2 * b * h3) * A * B

    return fu1 - fu2 + fu3


def G(a=a, b=b, c=c, u1=u1, u2=u2, u3=u3, h1=h1, h2=h2, h3=h3):
    A = (h1 + a) ** 2
    B = (h2 + b - a) ** 2
    C = (h3 - b) ** 2

    fu1 = (c - u1) ** 2 * a ** 3 / A
    fu2 = (c - u2) ** 2 * (b - a) ** 3 / B
    fu3 = (c - u3) ** 2 * b ** 3 / C

    return fu1 + fu2 - fu3


# Construct F (or G) as a quadratic in c and solve for c+-


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

def f1(a=a, b=b, c=c, s=s, u1=u1, u2=u2, u3=u3, h1=h1, h2=h2, h3=h3):
    A = (h1 + a) ** 2
    B = (h2 + b - a) ** 2

    fu1 = (c - u1) ** 2 * (1 - (h1 / A) ** 2)
    fu2 = (c - u2) ** 2 * (1 - (h2 / B) ** 2)

    return fu1 - fu2 - 2 * a / s


def f2(a=a, b=b, c=c, s=s, u1=u1, u2=u2, u3=u3, h1=h1, h2=h2, h3=h3):
    B = (h2 + b - a) ** 2
    C = (h3 - b) ** 2

    fu2 = (c - u2) ** 2 * (1 - (h2 / B) ** 2)
    fu3 = (c - u3) ** 2 * (1 - (h3 / C) ** 2)

    return fu2 - fu3 - 2 * b


def f3(a=a, b=b, c=c, s=s, u1=u1, u2=u2, u3=u3, h1=h1, h2=h2, h3=h3):
    A = (h1 + a) ** 2
    B = (h2 + b - a) ** 2
    C = (h3 - b) ** 2

    fu1 = (c - u1) ** 2 * a ** 3 / A ** 2
    fu2 = (c - u2) ** 2 * (b - a) ** 3 / B ** 2
    fu3 = (c - u3) ** 2 * b ** 3 / C ** 2

    return fu1 + fu2 - fu3


class Example_nsolve(object):
    def __init__(self, s=s, u1=u1, u2=u2, u3=u3, h1=h1, h2=h2, h3=h3):
        """Solve the base equations for (a, b, c) using sympy's
        nsolve. All the arguments need to be specified."""
        self.eq1 = f1(s=s, u1=u1, u2=u2, u3=u3, h1=h1, h2=h2, h3=h3)
        self.eq2 = f2(s=s, u1=u1, u2=u2, u3=u3, h1=h1, h2=h2, h3=h3)
        self.eq3 = f3(s=s, u1=u1, u2=u2, u3=u3, h1=h1, h2=h2, h3=h3)

        self.equation_set = (self.eq1, self.eq2, self.eq3)
        self.variables = (a, b, c)

    def solve(self, guess, **kwargs):
        return sp.nsolve(self.equation_set, self.variables, guess, **kwargs)


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
