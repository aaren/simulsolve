When a two layer fluid flows over topography, there exist a number
of different solutions depending on the relative size of the flow
speed and the wave speed in the fluid.

![critical-flow][1]

These are termed 'supercritical', 'subcritical' and 'critical' (the
first two I refer to here as 'extra-critical').

The following equations define the bounding lines between critical
and extra-critical behaviour in (h, U0) parameter space:

![eq1][eq1]

![eq2][eq2]


I want to eliminate d_1c and find solutions to these equations in (h, U_0).

Simplifying factors:

- I only need answers for *given* d_0
- *I do not need exact solutions*, just an outline of the solution
  curves, so this can be solved either analytically or numerically.
- I only want to plot over the region (h, U0) = (0,0) to (0.5, 1).

I'd like to solve this using modules available in the Enthought
distribuion (numpy, scipy, sympy), but really don't know where to
start. It's the elimination of the variable d1c that really confuses
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
always physical, but don't worry about that) and looks roughly
like this:

![critical-regime-diagram][2]

How do I go about implementing this?

  [eq1]: http://latex.codecogs.com/gif.latex?U_0%5E2%20%5Cleft%28%5Cfrac%7Bd_0%5E2%7D%7Bd_%7B1c%7D%5E3%7D%20&plus;%20%5Cfrac%7B%281%20-%20d_0%29%5E2%7D%7B%281%20-%20d_%7B1c%7D%20-%20d_0%20h%29%5E3%7D%20%5Cright%29%20-%201%20%3D%200
  [eq2]: http://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7B2%7D%20U_0%5E2%20%5Cleft%28%20%5Cfrac%7Bd_0%5E2%7D%7Bd_%7B1c%7D%5E2%7D%20&plus;%20%5Cfrac%7B%281%20-%20d_0%29%5E2%7D%7B%281%20-%20d_%7B1c%7D%20-%20d_0%20h%29%5E2%7D%20%5Cright%29%20&plus;%20d_%7B1c%7D%20&plus;%20d_0%20%28h%20-%201%29%20%3D%200  


  [1]: http://i.stack.imgur.com/DLeEN.png
  [2]: http://i.stack.imgur.com/egrNP.png
