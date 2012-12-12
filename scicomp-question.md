When a two layer fluid flows over topography, there exist a number
of different solutions depending on the relative size of the flow
speed and the wave speed in the fluid.

These are termed 'supercritical', 'subcritical' and 'critical' (the
first two I refer to here as 'extra-critical').

The following equations define the bounding lines between critical
and extra-critical behaviour in $(h, U_0)$ parameter space (adapted from Baines 1994, via White and Helfrich 2012, both in JFM):

\\[ U_0^2 \left(\frac{d_0^2}{d_{1c}^3} + \frac{(1 - d_0)^2}{(1 - d_{1c} - d_0 h)^3} \right) - 1 = 0 \\]

\\[ \frac{1}{2} U_0^2 \left( \frac{d_0^2}{d_{1c}^2} + \frac{(1 - d_0)^2}{(1 - d_{1c} - d_0 h)^2} \right) + d_{1c} + d_0 (h - 1) = 0 \\]

I want to eliminate $d_{1c}$ and find solutions to these equations in $(h, U_0)$.

Simplifying factors:

- I only need answers for *given* $d_0$
- *I do not need exact solutions*, just an outline of the solution
  curves, so this can be solved either analytically or numerically.
- I only want to plot over the region $(h, U_0) = (0,0) \rightarrow (0.5, 1)$.

I'd like to solve this using modules available in the Enthought
distribuion (numpy, scipy, sympy), but really don't know where to
start. It's the elimination of the variable $d_{1c}$ that really confuses
me.  

Here are the equations in python:

<!-- language: python -->

    def eq1(h, U0, d1c, d0=0.1):
        f = (U0) ** 2 * ((d0 ** 2 / d1c ** 3) + (1 - d0) ** 2 / (1 - d1c - d0 * h) ** 3) - 1
        return f

    def eq2(h, U0, d1c, d0=0.1):
        f = 0.5 * (U0) ** 2 * ((d0 ** 2 / d1c ** 2) + (1 - d0) ** 2 / (1 - d1c - d0 * h)) + d1c + d0 * (h - 1)
        return f

I'm expecting a solution that has a number of solution branches (not
always physical, but don't worry about that) and looks roughly like a
few lines and parabolas in $(h, U_0)$.

How do I go about implementing this?

Same question on [StackOverflow][so-q], complete with pictures.

[so-q]: http://stackoverflow.com/questions/13823275/  
