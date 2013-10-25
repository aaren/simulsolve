import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

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

        self.H = h1, h2, h3
        self.V = v1, v2, v3

    def insolutiontriangle(self, ab):
        """Boolean, is ab = (a, b) inside the physical solution
        triangle?
        """
        h1, h2, h3 = self.H

        a, b = ab

        c1 = (a > -h1) & (a < (1 - h1))
        c2 = (b > (a - h2)) & (b < (a + 1 - h2))
        c3 = (b > -(h1 + h2)) & (b < h3)

        return c1 & c2 & c3

    def rough_zero_ordered(self):

        a, b = self.global_rough_zero(self.f)
        f0 = np.column_stack((a, b))

        segments = {}
        # select upper right quadrant
        segments['upper_right'] = f0[np.where((a > 0) & (b > 0))]
        # select upper left quadrant
        segments['upper_left'] = f0[np.where((a < 0) & (b > 0))]
        # select lower left quadrant
        segments['lower_left'] = f0[np.where((a < 0) & (b < 0))]
        # select lower right quadrant
        segments['lower_right'] = f0[np.where((a > 0) & (b < 0))]

        # for the origin segments, sort by proximity to origin
        def sort_origin(points):
            distance_from_origin = np.sum(points ** 2, axis=1)
            return points[np.argsort(distance_from_origin)]

        for quad in segments:
            segments[quad] = sort_origin(segments[quad])

        # for the upper left corner segment, take lower left
        # and remove points outside the corner (-h1, h3), then
        # sort by proximity to the corner
        ul = segments['upper_left']
        # use or here because we want to keep the rest of
        # the curve so that we can terminate when it goes
        # back outside the solution triangle.
        h1, h2, h3 = self.H
        # move the corner in a bit to ignore the return branch
        # bit hacky but it works. There must be some extreme
        # parameters that can be out here.
        h1 -= 0.01
        h3 -= 0.01
        ulc = ul[np.where((ul[:, 0] > -h1) | (ul[:, 1] < h3))]
        distance_from_corner = np.hypot(ulc[:, 0] + h1, ulc[:, 1] - h3)
        ulc = ulc[np.argsort(distance_from_corner)]
        segments['upper_left_corner'] = ulc

        def kdt_sort(points):
            # create a kd-tree
            from scipy.spatial import KDTree
            kdt = KDTree(points.copy())

            # walk the tree (the list of points), finding the three
            # nearest points. The nearest is the point itself, so select
            # the next nearest as long as it isn't the last point found

            # starting point, at some known extreme
            indices = [0]
            while True:
                point = kdt.data[indices[-1]]
                distances, (i0, i1) = kdt.query(point, 2)
                # break if exhausted the points or next point
                # would be outside the solution triangle
                if i1 == kdt.data.shape[0]:
                    break
                next_point = kdt.data[i1]
                if not self.insolutiontriangle(next_point):
                    break
                # eliminate this index from consideration
                kdt.data[i0] *= np.nan
                indices.append(i1)
            return points[indices]

        ordered_segments = {}

        for segment in segments:
            points = segments[segment]
            ordered_points = kdt_sort(points)
            ordered_segments[segment] = ordered_points

        return ordered_segments

    def global_rough_zero(self, f):
        x = np.linspace(-1, 1, 1000)
        y = np.linspace(-1, 1, 1000)
        A, B = np.meshgrid(x, y)
        fab = f(A, B)
        # intersection between contour sets is done here:
        # http://stackoverflow.com/questions/17416268/
        # but we'll just use the bit for getting points
        # from matplotlib contour
        contour = plt.contour(A, B, fab, levels=[0])
        points = np.row_stack(p.vertices for line in contour.collections
                              for p in line.get_paths())
        a, b = points.T
        return a, b

    def zeroG(self, f0):
        """Evaluate G along the lines F=0.

        We are looking G=0, i.e. a sign change.

        Inputs:
            f0 - an array of (a, b), shape (N, 2)

        Return (a, b) in the vicinity of the sign change.
        """
        g = self.g(*f0.T)
        # find zero crossing
        # TODO: more elegant? like interpolating?
        zero = np.where(np.diff(np.sign(g)) != 0)
        ab = np.vstack((f0[zero], f0[zero])).squeeze()
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
        try:
            ab = sp.nsolve(equation_set, variables, guess, **kwargs)
        except:
            return guess
        return np.array(ab, dtype=float)

    @property
    def roots(self):
        """Calculate the roots of the system."""
        segments = self.rough_zero_ordered()

        roughzeros = [self.zeroG(segments[s]) for s in segments]
        # remove duplicates and concatenate
        guesses = np.row_stack(np.unique(g).reshape((-1, 2))
                               for g in roughzeros if g.size != 0)
        enhanced_guesses = [self.enhance(guess) for guess in guesses]

        return enhanced_guesses


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
