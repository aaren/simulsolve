Semi-formally, the problem you are trying to solve is the following:
given d0, solve the logical formula "there exists d1c such that
eq1(h, U0, d1c, d0) = eq2(h, U0, d1c, d0) = 0" for h and U0.

There exists an algorithm to reduce the formula to a polynomial
equation "P(h, U0) = 0", it's called "quantifier elimination" and it
usually relies on another algorithm, "cylindrical algebraic
decomposition". Unfortunately, this isn't implemented in sympy
(yet).

However, since U0 can easily be eliminated, there are things you can
do with sympy to find your answer. Start with
    
    h, U0, d1c, d0 = symbols('h, U0, d1c, d0')
    f1 = (U0) ** 2 * ((d0 ** 2 / d1c ** 3) + (1 - d0) ** 2 / (1 - d1c - d0 * h) ** 3) - 1
    f2 = U0**2 / 2 * ((d0 ** 2 / d1c ** 2) + (1 - d0) ** 2 / (1 - d1c - d0 * h)) + d1c + d0 * (h - 1)

Now, eliminate U0 from f1 and insert the value in f2 (I'm doing it
"by hand" rather than with solve() to get a prettier expression):

    U2_val = ((f1 + 1)/U0**2)**-1
    f3 = f2.subs(U0**2, U2_val)

f3 only depends on h and d1c. Also, since it's a rational fraction,
we only care about when its numerator goes to 0, so we get a single
polynomial equation in 2 variables:

    p3 = fraction(cancel(f3))

Now, for a given d0, you should be able to invert p3.subs(d0, .1)
numerically to get h(d1c), plug it back into U0 and make a
parametric plot of (h, U0) as a function of d1c.
