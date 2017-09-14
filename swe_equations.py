from sympy import symbols, Derivative, Eq, pretty_print

u, v, h, t, x, y, g, H, f = symbols('u v h t x y g H f')

eq1 = Eq(Derivative(u, t) + g * Derivative(h, x) - f * v, 0)
eq2 = Eq(Derivative(v, t) + g * Derivative(h, y) + f * u, 0)
eq3 = Eq(Derivative(h, t) + H * (Derivative(u, x) + Derivative(v, y)), 0)

pretty_print(eq1)
pretty_print(eq2)
pretty_print(eq3)
