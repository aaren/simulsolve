Hi all,

First post. I'm a PhD student at the University of Leeds, looking at
the fundamental behaviour of storm systems in the atmosphere.

I'm trying to solve two simultaneous polynomial equations, that
together define the boundaries of a region in the parameter space of 
a fluid mechanics problem.

Latex:
$$ U_0^2 \left(\frac{d_0^2}{d_{1c}^3} + \frac{(1 - d_0)^2}{(1 - d_{1c} - d_0 h)^3} \right) - 1 = 0 $$
$$ 0.5 U_0^2 \left( \frac{d_0^2}{d_{1c}^2} + \frac{(1 - d_0)^2}{(1 - d_{1c} - d_0 h)^2} \right) + d_{1c} + d_0 (h - 1) = 0 $$

Python:
eq1 = (U0) ** 2 * ((d0 ** 2 / d1c ** 3) + (1 - d0) ** 2 / (1 - d1c - d0 * h) ** 3) - 1
eq2 = 0.5 * (U0) ** 2 * ((d0 ** 2 / d1c ** 2) + (1 - d0) ** 2 / (1 - d1c - d0 * h)) + d1c + d0 * (h - 1)

I want to eliminate d1c and find solutions in (h, U0) for given d0.

I just need enough data to plot the solution curves in (h, U0). 
This could mean exact solutions, or an approximate polynomial that 
describes a solution branch or an array of roots over a grid.

If it helps I also only need to find solutions in the region 
(0,0) -> (0.5, 1) in (h, U0).

I don't really know where to start. The elimination of d1c puzzles me.

I'm using sympy 0.7.2 with Enthought 7.3-2.

I've asked the same question on [StackOverflow][so-q], where there 
are typeset equations, pictures and more on the physical problem.

[so-q]: http://stackoverflow.com/questions/13823275/

Is there a neat way to solve this, numerically or otherwise?

Thanks,
Aaron
