from profilehooks import profile

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

#### Global symbols used for symby manipulations
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
####


def unique(a):
    """Remove duplicates from a 2d array
    http://stackoverflow.com/questions/8560440
    """
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return a[ui]


class FGSolver(object):
    """For given vi, hi, s, find the possible solutions in (a, b).

    This solver uses the root finding method described in Lamb 2000,
    which is to assume a relation between the fluid velocities and
    the wave speed of the root,

        c * Vi = c - Ui.

    This assumption allows us to reduce the equations from three in
    (a, b, c) to two in (a, b). A graphical method is then used to
    find the roots by searching along the contours F(a, b) = 0 for
    G(a, b) = 0.

    A constraint on the allowed physical values of hi allows us to
    separate these roots (see self.insolutiontriangle).
    """
    def __init__(self, s=1, h1=0.2, h2=0.6, h3=0.2, v1=1, v2=1, v3=1):
        """Inputs: s  - the stratification parameter
                   h1 - depth of layer 1
                   h2 - depth of layer 2
                   h3 - depth of layer 3
                   v1 - speed relation for layer 1
                   v2 - speed relation for layer 2
                   v3 - speed relation for layer 3.
        """
        # Create functions of (a, b) for these given parameters
        self.F = self.F(v1=v1, v2=v2, v3=v3, h1=h1, h2=h2, h3=h3, s=s)
        self.G = self.G(v1=v1, v2=v2, v3=v3, h1=h1, h2=h2, h3=h3)
        self.f = sp.lambdify((a, b), self.F)
        self.g = sp.lambdify((a, b), self.G)

        self.H = h1, h2, h3
        self.V = v1, v2, v3

        self.base = LambBase(s=s, h1=h1, h2=h2, h3=h3)

        # resolution of rough zero search (np.linspace)
        self.resolution = 1000

    @staticmethod
    def F(a=a, b=b, v1=v1, v2=v2, v3=v3, h1=h1, h2=h2, h3=h3, s=s):
        A = (h1 + a) ** 2
        B = (h2 + b - a) ** 2
        C = (h3 - b) ** 2

        f1 = - (a + b * s) * v2 ** 2 * h2 ** 2 * A * C
        f2 = B * (a * v3 ** 2 * h3 ** 2 * A + b * s * v1 ** 2 * h1 ** 2 * C)
        f3 = A * B * C * (a * (v2 ** 2 - v3 ** 2)
                          + b * s * (v2 ** 2 - v1 ** 2))

        return f1 + f2 + f3

    @staticmethod
    def G(a=a, b=b, v1=v1, v2=v2, v3=v3, h1=h1, h2=h2, h3=h3):
        A = (h1 + a) ** 2
        B = (h2 + b - a) ** 2
        C = (h3 - b) ** 2

        fu1 = v1 ** 2 * a ** 3 / A
        fu2 = v2 ** 2 * (b - a) ** 3 / B
        fu3 = v3 ** 2 * b ** 3 / C

        return fu1 + fu2 - fu3

    def insolutiontriangle(self, (a, b)):
        """Boolean, is ab = (a, b) inside the physical solution
        triangle?
        """
        h1, h2, h3 = self.H

        c1 = (a > -h1) & (a < (1 - h1))
        c2 = (b > (a - h2)) & (b < (a + 1 - h2))
        c3 = (b > -(h1 + h2)) & (b < h3)

        return c1 & c2 & c3

    def rough_zero_ordered(self):
        """The zero contours of the function F(a, b) = 0 can be
        separated distinctly in the (a, b) plane.

        This method calculates roughly where the contours are
        and then splits them up into points.

        Each segment describes a line in (a, b). The points in each
        segment are sorted such that they are ordered along this
        line starting from one end.
        """
        # extract the rough locations of the zeros
        a, b = self.global_rough_zero(self.f)
        f0 = np.column_stack((a, b))

        points = {}
        # select upper right quadrant
        points['upper_right'] = f0[np.where((a > 0) & (b > 0))]
        # select upper left quadrant
        points['upper_left'] = f0[np.where((a < 0) & (b > 0))]
        # select lower left quadrant
        points['lower_left'] = f0[np.where((a < 0) & (b < 0))]
        # select lower right quadrant
        points['lower_right'] = f0[np.where((a > 0) & (b < 0))]

        # for the origin points, sort by proximity to origin
        def sort_origin(points):
            distance_from_origin = np.sum(points ** 2, axis=1)
            return points[np.argsort(distance_from_origin)]

        for quad in points:
            points[quad] = sort_origin(points[quad])

        # for the upper left corner segment, take lower left
        # and remove points outside the corner (-h1, h3), then
        # sort by proximity to the corner
        ul = points['upper_left']
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
        points['upper_left_corner'] = ulc

        @profile
        def kdt_sort(points):
            """Sort the set of given points along a line by minimising
            the point to point distance. This forms a minimum spanning
            path for the set of points, that visits each point exactly
            once.

            Requires the point at index 0 to be at one end of the
            line.
            """
            # create a kd-tree
            from scipy.spatial import KDTree
            # copy so we don't change the input data
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

        ordered_segments = {segment: kdt_sort(points[segment])
                            for segment in points}

        return ordered_segments

    def global_rough_zero(self, f):
        x = np.linspace(-1, 1, self.resolution)
        A, B = np.meshgrid(x, x)
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
        # you could create a kdtree with the F contour points
        # and one with G, then find nearest neighbours
        # e.g. kdt = KDTree(f_points)
        # kdt.query(g_points, k=1, distance_upper_bound=something)
        # this will pick up curves that very nearly intersect
        # but not quite - although this is a very particular edge
        # case.

    def zeroG(self, f0):
        """Evaluate G along the lines F=0.

        We are looking G=0, i.e. a sign change.

        Inputs:
            f0 - an array of (a, b), shape (N, 2)

        Return (a, b) in the vicinity of the sign changes.
        """
        g = self.g(*f0.T)
        # find zero crossings
        # TODO: more elegant? like interpolating?
        zero = np.where(np.diff(np.sign(g)) != 0)
        ab = np.vstack((f0[zero], f0[zero])).squeeze()
        return ab

    def enhance(self, guess, **kwargs):
        """Using a rough guess for (a, b), converge on the
        zero using a non linear solver with a newton raphson
        method.

        Input: guess - a numpy array of floats [a, b]

        Output: a numpy array of floats [a, b]
                OR an empty array if no root can be found.
        """
        # nsolve can't take arrays as input for some reason
        guess = guess.tolist()
        variables = (a, b)
        equation_set = (self.F, self.G)
        try:
            ab = sp.nsolve(equation_set, variables, guess, **kwargs)
        except ValueError:
            ab = []
        return np.array(ab, dtype=float)

    @property
    def roots(self):
        """Calculate the roots of the system.

        Returns an array of roots in (a, b, c):

            [(a0, b0, c0),
             (a1, b1, c1),
             ...]

        """
        if not hasattr(self, '_roots'):
            roots = self.calculate_roots()
            setattr(self, '_roots', roots)
        return self._roots

    def calculate_roots(self):
        """Compute the roots of the system, excluding trivial zero
        solutions.

        Returns an array of roots in (a, b, c):

            [(a0, b0, c0),
             (a1, b1, c1),
             ...]

        N.B. the value for c is the *magnitude* only.
        """
        # compute the rough zero contours of F
        zero_contours = self.rough_zero_ordered().values()
        # find where G becomes zero on each branch in each quadrant
        roughroots = np.row_stack(self.zeroG(branch)
                                  for branch in zero_contours)
        # remove duplicates
        guesses = unique(roughroots)
        # find exact solutions near each of the guesses
        enhanced_roots = [self.enhance(guess) for guess in guesses]
        # reject roots that didn't converge
        enhanced_roots = np.row_stack(r for r in enhanced_roots if len(r) > 0)
        # remove trivial zero solutions
        res = 2. / self.resolution
        nonzero_enhanced_roots = np.row_stack(g for g in enhanced_roots
                                                if np.hypot(*g) > res)
        # compute c for each of the roots
        C = self.compute_c(nonzero_enhanced_roots.T)
        abc_roots = np.column_stack((nonzero_enhanced_roots, C))

        return abc_roots

    def compute_c(self, (a, b)):
        """Given values of (a, b), calculate
        the *magnitude* of the wave speed.

        Uses the relation

        c * Vi = c - Ui

        and

        (c - Ui) ** 2 = f(a, b)

        =>  c ** 2 = f(a, b) / Vi ** 2


        """
        # we use v1 here, but could use any of the layers
        fcu1 = self.base.fcu1
        v1 = self.V[0]

        c_squared = fcu1(a, b) / v1 ** 2

        return c_squared ** .5

    def compute_U(self, (a, b, c)):
        """For given a, b, c calculate the velocities ui."""
        u1 = c - self.base.fcu1(a, b) ** .5
        u2 = c - self.base.fcu2(a, b) ** .5
        u3 = c - self.base.fcu3(a, b) ** .5
        return u1, u2, u3

    @property
    def Uroots(self):
        """For each root, compute the velocities ui."""
        return np.array(self.compute_U(self.roots.T)).T


class GivenUSolver(object):
    """For given Ui, hi, s determine solutions in (a, b, c)."""
    def __init__(self, u1=0, u2=0, u3=0, h1=0.2, h2=0.6, h3=0.2, s=1):
        """Inputs: s  - the stratification parameter
                   h1 - depth of layer 1
                   h2 - depth of layer 2
                   h3 - depth of layer 3
                   u1 - speed of layer 1
                   u2 - speed of layer 2
                   u3 - speed of layer 3.
        """
        self.U = u1, u2, u3
        self.H = h1, h2, h3
        self.s = s

        # Calculate upper and lower bounds on c.
        #
        # In a three layer fluid at rest, the maximum conjugate
        # speed is equal to the conjugate speed in the two layer
        # system formed by the top and bottom layers.
        # If we *assume* that this is also true for fluids not at
        # rest, then we can set bounds on where solutions can exist.
        #
        # TODO: really need to check this assumption! it seems odd
        # that the bounds wouldn't depend at all on the speed of
        # the middle layer.
        #
        # We can express the reduced gravity of the upper and lower
        # layers as g'13 = g'12 + g'23.
        # Now, as we scale by g'23 in the non-dimensionalisation, we
        # obtain g'13 = 1 + 1 / s
        g13 = (1 + s) / s

        # the absolute bound on the rightward propagating wave
        # (direction Ui+) is c_r = g13 ** .5 + u1
        # and this occurs when |u3 - u1| = sqrt((1 + s) / s).
        # if |u3 - u1| exceeds this bound, there are *no solutions*.
        if np.abs(u3 - u1) >= g13 ** .5:
            print "There are probably no solutions for these speeds!"
            print "Limiting layer speed difference is %s" % g13 ** .5
        # When |u3 - u1| < sqrt((1 + s) / s), the rightward
        # propagating wave in the two layer system has speed
        c_r = 0.5 * ((u3 - u1) + g13 ** .5) + u1
        # similarly the leftward wave has speed
        c_l = -0.5 * ((u1 - u3) + g13 ** .5) + u1

        # We now *assume* that these represent the *bounds* on the
        # wave speed in the three layer system.
        self.c_hi = c_r
        self.c_lo = c_l
        print "The bounds on the wave speeds are:"""
        print (c_l, c_r)

        self.scan_res = 20
        self.Cg = np.linspace(self.c_lo, self.c_hi, self.scan_res)

    def solver_given_c(self, c=1):
        """For a given value of c, compute the vi that correspond to
        the given ui and create a solver with these vi.

        The value of c used (cg) corresponds to the speed of a root
        iff one of the roots found by the solver has a root speed
        cr = cg.

        In practice we are unlikely to hit the root exactly, but we
        hope that the roots are sufficiently separated in (a, b, c)
        that we can differentiate them as long as we guess cg with
        sufficient resolution.
        """
        u1, u2, u3 = self.U
        h1, h2, h3 = self.H
        v1 = 1 - u1 / c
        v2 = 1 - u2 / c
        v3 = 1 - u3 / c

        solver = FGSolver(s=self.s, v1=v1, v2=v2, v3=v3, h1=h1, h2=h2, h3=h3)
        return solver

    def scan_c(self, Cg=None):
        C = Cg or self.Cg
        solvers = [self.solver_given_c(c) for c in C]
        roots = [s.roots for s in solvers]
        velocities = [s.Uroots for s in solvers]

        def append_c(array, c):
            """Put a single value on the end of each row in 2d array."""
            c_shape = (array.shape[0], 1)
            C = np.ones(c_shape) * c
            return np.hstack((array, C))

        RC = np.row_stack(append_c(r, c) for r, c in zip(roots, C))
        UC = np.row_stack(append_c(r, c) for r, c in zip(velocities, C))

        return RC, UC

    def root_find(self):
        RC, UC = self.scan_c()
        a, b, cr, cg = RC.T

        # this value depends on the resolution with which we scan
        # over c
        # C = np.linspace(lo, hi, res)
        # spacing = (hi - lo) / res
        # e.g. (2 - -2) / 100 = 0.04
        # then we need to consider spacing either side of cr
        # so brackets of at least 2 * spacing to guarantee
        # finding two points that span 1
        r = 2 * (self.c_hi - self.c_lo) / self.scan_res

        # limit data to values that are near to a root (cr = cg)
        diff = np.abs(cr - np.abs(cg))
        close = np.where(diff < r)
        data = RC[close]

        # extract from limited data
        a, b, cr, cg = data.T

        # Now we project the data so that the roots are well
        # separated

        # angle that the roots make with +a axis in the (a, b) plane
        x = np.arctan2(b, a)

        # split into positive and negative root velocities.
        # we do this using the fact that cr is a magnitude.
        y_pos = cr - cg
        y_neg = cr + cg

        # stick all the points together. This is now a re-projection
        # of the original data that is near to a root into a space
        # where the roots are separated.
        XY_pos = np.column_stack((x, y_pos))
        XY_neg = np.column_stack((x, y_neg))

        # use a density based sort to separate points into distinct
        # clusters, each one corresponding to a distinct root.
        # There is an edge case here in that roots can become
        # degenerate: as the roots come closer together, they will
        # be treated as a single root cluster at some non-zero
        # separation distance.
        from sklearn.cluster import DBSCAN

        db_pos = DBSCAN(eps=r, min_samples=2)
        db_pos.fit(XY_pos)

        db_neg = DBSCAN(eps=r, min_samples=2)
        db_neg.fit(XY_neg)

        max_label = int(db_pos.labels_.max())
        labels = range(max_label + 1)
        branch_pos = [XY_pos[np.where(db_pos.labels_ == label)] for label in labels]
        data_pos = [data[np.where(db_pos.labels_ == label)] for label in labels]

        max_label = int(db_neg.labels_.max())
        labels = range(max_label + 1)
        branch_neg = [XY_neg[np.where(db_neg.labels_ == label)] for label in labels]
        data_neg = [data[np.where(db_neg.labels_ == label)] for label in labels]

        # First, select only the branches that contain a sign change.
        def has_sign_change(branch_data):
            x, y = branch_data.T
            change = np.sum(np.abs(np.diff(np.sign(y))))
            return bool(change)

        branches_pos = [data for branch, data in zip(branch_pos, data_pos) if has_sign_change(branch)]
        branches_neg = [data for branch, data in zip(branch_neg, data_neg) if has_sign_change(branch)]

        def get_guesses(branches):
            guesses = []
            for branch in branches:
                a, b, cr, cg = branch.T
                guess = branch[np.argmin(np.abs(cr - np.abs(cg)))]
                guesses.append(guess)
            return guesses

        guesses_pos = get_guesses(branches_pos)
        guesses_neg = get_guesses(branches_neg)

        guess_array = np.array(guesses_pos + guesses_neg)

        s = self.s
        u1, u2, u3 = self.U
        h1, h2, h3 = self.H
        # now find the exact solutions
        lambsolver = LambBaseSolver(s=s, h1=h1, h2=h2, h3=h3,
                                    u1=u1, u2=u2, u3=u3)

        a, b, cr, cg = guess_array.T
        # find the correct sign of cr by using the sign of cg
        guesses = np.column_stack((a, b, cr * np.sign(cg)))
        solutions = np.row_stack(lambsolver.solve(guess) for guess in guesses)

        return solutions


class LambBase(object):
    """Lamb 2000, base equation set, in the Boussinesq limit.

    *These are NON-DIMENSIONALISED*, i.e. you have to input
    non-dimensional parameters.

        lengths
            scale with H

        accelerations (i.e. gravity)
            scale with g' = N_2^2 = g * (rho_2 - rho_3) / rho_2

        velocities
            scale with (g' * H) ^ 0.5

    If you want the non-Boussinesq equations, you'll have to
    rewrite, dimensionalising everything as well.
    """
    def __init__(self, s=s, h1=h1, h2=h2, h3=h3):
        """An instance initialised with numerical values will
        allow calculation of c - ui as a function of (a, b).
        """
        # Create some useful relations.
        # These are the perturbed layer depths.
        A = (h1 + a)
        B = (h2 + b - a)
        C = (h3 - b)

        self.Ai = A, B, C

        H1 = A ** 2 - h1 ** 2
        H2 = B ** 2 - h2 ** 2
        H3 = C ** 2 - h3 ** 2

        self.Hi = H1, H2, H3

        self.alpha = 2 * a / s
        self.beta = 2 * b

    @staticmethod
    def f1(a=a, b=b, c=c, s=s, u1=u1, u2=u2, u3=u3, h1=h1, h2=h2, h3=h3):
        """Lamb's equation 12a in the Boussinesq limit."""
        A = (h1 + a)
        B = (h2 + b - a)

        fu1 = (c - u1) ** 2 * (1 - (h1 / A) ** 2)
        fu2 = (c - u2) ** 2 * (1 - (h2 / B) ** 2)

        return fu1 - fu2 - 2 * a / s

    @staticmethod
    def f2(a=a, b=b, c=c, s=s, u1=u1, u2=u2, u3=u3, h1=h1, h2=h2, h3=h3):
        """Lamb's equation 12b in the Boussinesq limit."""
        B = (h2 + b - a)
        C = (h3 - b)

        fu2 = (c - u2) ** 2 * (1 - (h2 / B) ** 2)
        fu3 = (c - u3) ** 2 * (1 - (h3 / C) ** 2)

        return fu2 - fu3 - 2 * b

    @staticmethod
    def f3(a=a, b=b, c=c, s=s, u1=u1, u2=u2, u3=u3, h1=h1, h2=h2, h3=h3):
        """Lamb's equation 14 in the Boussinesq limit."""
        A = (h1 + a)
        B = (h2 + b - a)
        C = (h3 - b)

        fu1 = (c - u1) ** 2 * a ** 3 / A ** 2
        fu2 = (c - u2) ** 2 * (b - a) ** 3 / B ** 2
        fu3 = (c - u3) ** 2 * b ** 3 / C ** 2

        return fu1 + fu2 - fu3

    @property
    def cu1(self):
        """Rearrangment of f1, f2, f3 to obtain (c - U1) ** 2 as a
        function of (a, b, hi, s)
        """
        H1, H2, H3 = self.Hi
        A, B, C = self.Ai
        alpha, beta = self.alpha, self.beta

        n = H3 * alpha * (b - a) ** 3 - H2 * (alpha + beta) * b ** 3
        d = H2 * H3 * a ** 3 + H1 * H3 * (b - a) ** 3 - H1 * H2 * b ** 3
        return A ** 2 * n / d

    @property
    def cu2(self):
        """Rearrangment of f1, f2, f3 to obtain (c - U2) ** 2 as a
        function of (a, b, hi, s)
        """
        H1, H2, H3 = self.Hi
        A, B, C = self.Ai
        alpha, beta = self.alpha, self.beta

        return B ** 2 * (H1 * (self.cu1 / A ** 2) - alpha) / H2

    @property
    def cu3(self):
        """Rearrangment of f1, f2, f3 to obtain (c - U3) ** 2 as a
        function of (a, b, hi, s)
        """
        H1, H2, H3 = self.Hi
        A, B, C = self.Ai
        alpha, beta = self.alpha, self.beta

        return C ** 2 * (H1 * (self.cu1 / A ** 2) - (alpha + beta)) / H3

    @property
    def fcu1(self):
        """Create a function f(a, b) = (c - U1) ** 2.

        """
        return sp.lambdify((a, b), self.cu1)

    @property
    def fcu2(self):
        """Create a function f(a, b) = (c - U2) ** 2.

        """
        return sp.lambdify((a, b), self.cu2)

    @property
    def fcu3(self):
        """Create a function f(a, b) = (c - U3) ** 2.

        """
        return sp.lambdify((a, b), self.cu3)


class LambBaseSolver(object):
    def __init__(self, s=s, u1=u1, u2=u2, u3=u3, h1=h1, h2=h2, h3=h3):
        """Solve the base equations for (a, b, c) using sympy's
        nsolve. All hi, Ui, s need to be given.

        The solve method of this class takes an initial guess and finds the
        closest root of the three equation system. Your guess needs to be
        pretty close to the root or this may not converge. Thus this class
        is only useful for root refinement.
        """
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3

        self.eq1 = LambBase.f1(s=s, u1=u1, u2=u2, u3=u3, h1=h1, h2=h2, h3=h3)
        self.eq2 = LambBase.f2(s=s, u1=u1, u2=u2, u3=u3, h1=h1, h2=h2, h3=h3)
        self.eq3 = LambBase.f3(s=s, u1=u1, u2=u2, u3=u3, h1=h1, h2=h2, h3=h3)

        self.equation_set = (self.eq1, self.eq2, self.eq3)
        self.variables = (a, b, c)

    def solve(self, guess, **kwargs):
        guess = tuple(guess)
        try:
            abc = sp.nsolve(self.equation_set, self.variables, guess, **kwargs)
            abc = np.array(abc, dtype=np.float)
        except ValueError:
            # non convergence of solver
            abc = None
        return abc


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
if __name__ == '__main__':
    usolver = GivenUSolver(s=0.5, h1=0.2, h2=0.35, h3=0.45, u1=0, u2=-0.4, u3=-0.1)
    solutions = usolver.root_find()
    print solutions
