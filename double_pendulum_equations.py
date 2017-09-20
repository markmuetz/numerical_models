import pylab as plt

from sympy import symbols, Derivative, Eq, cos, diff
from sympy import simplify, solve
from sympy import pretty_print
from sympy.physics.vector import dynamicsymbols


def double_pendulum():
    th1, th2 = dynamicsymbols('theta_1 theta_2')
    t, m, g, l = symbols('t m g l')

    om1 = diff(th1, t)
    om2 = diff(th2, t)

    # Equations for KE
    v1 = l * om1
    v2 = l * (om1 + cos(th1 - th2) * om2)
    T = 1. / 2. * m * (v1**2 + v2**2)

    # Equations for PE.
    h1 = l * (1 - cos(th1))
    h2 = h1 - l * cos(th2)
    V = m * g * (h1 + h2)

    # Lagrangian
    L = T - V

    #pretty_print(T)
    #pretty_print(V)
    #pretty_print(L)

    dL_dth1 = diff(L, th1)
    dL_dth2 = diff(L, th2)
    #pretty_print(dL_dth1)

    dL_dom1 = diff(L, om1)
    dL_dom2 = diff(L, om2)

    dt1 = diff(dL_dom1, t)
    dt2 = diff(dL_dom2, t)

    eq1 = simplify(Eq(dt1, dL_dth1))
    eq2 = simplify(Eq(dt2, dL_dth2))

    pretty_print(eq1)
    pretty_print(eq2)
    return eq1, eq2

def simple_pendulum():
    th = dynamicsymbols('theta')
    t, m, g, l = symbols('t m g l')
    om = diff(th, t)
    omdot = diff(om, t)

    # Equation for KE.
    v = l * om
    T = 1. / 2. * m * (v**2)

    # Equation for PE.
    h1 = l * (1 - cos(th))
    V = m * g * (h1)

    # Lagrangian
    L = T - V

    dL_dth = diff(L, th)
    dL_dom = diff(L, om)

    dt = diff(dL_dom, t)
    eq = simplify(Eq(dt, dL_dth))
    eq = Eq(omdot, solve(eq, omdot)[0])
    pretty_print(eq)
    return th, om, omdot, t, m, g, l, eq

if __name__ == '__main__':
    th, om, omdot, t, m, g, l, eq = simple_pendulum()

    # Discretize equations.
    dt = symbols('Delta')
    om2, om1 = symbols('omega_2 omega_1')
    th2, th1 = symbols('theta_2 theta_1')
    sol_eq1 = Eq(om2, solve(eq.subs(omdot, (om2 - om1) / dt), om2)[0])
    sol_eq1 = sol_eq1.subs({th: th1})
    sol_eq2 = Eq(th2, solve(Eq((th2 - th1) / dt, om1), th2)[0])
    pretty_print(sol_eq1)
    pretty_print(sol_eq2)

    # Solve equations (in the least efficient way possible!).
    om0 = 0
    th0 = 3
    nt = 1000
    om2_val = solve(sol_eq1.subs({g:9.81, l:1, dt:0.01, om1:0, th1:3}).evalf(), om2)[0]
    th2_val = solve(sol_eq2.subs({th1:3, dt:0.01, om1:0}), th2)[0]
    omegas = [om2_val]
    thetas = [th2_val]

    for n in range(nt):
        print(n)
        om2_val = solve(sol_eq1.subs({g:9.81, l:1, dt:0.01, om1:om2_val, th1:th2_val}).evalf(), om2)[0]
        th2_val = solve(sol_eq2.subs({th1:th2_val, dt:0.01, om1:om2_val}), th2)[0]
        omegas.append(om2_val)
        thetas.append(th2_val)

    plt.plot(thetas)
