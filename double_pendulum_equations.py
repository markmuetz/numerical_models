import pylab as plt
import numpy as np

from sympy import symbols, Derivative, Eq, cos, diff
from sympy import simplify, solve
from sympy import pretty_print
from sympy.physics.vector import dynamicsymbols

def plot_double_pendulum(l1, l2, th1s, th2s):
    for th1, th2 in zip(th1s[::50], th2s[::50]):
        plt.clf()
        plt.xlim(-(l1 + l2), l1 + l2)
        plt.ylim(-(l1 + l2), l1 + l2)
        x1 = l1 * np.sin(th1)
        y1 = -l1 * np.cos(th1)
        x2 = l1 * np.sin(th1) + l2 * np.sin(th2)
        y2 = -l1 * np.cos(th1) - l2 * np.cos(th2)
        plt.plot([0, x1], [0, y1], 'k-')
        plt.plot(x1, y1, 'ro')
        plt.plot([x1, x2], [y1, y2], 'k-')
        plt.plot(x2, y2, 'bo')
        plt.pause(0.001)

def double_pendulum_impl(dt=0.001, th1_0=1.3, th2_0=0.3, om1_0=0,  om2_0=0,  m1=1.,  m2=1.,  l1=1.,  l2=1.,  g=9.81):
    th1 = th1_0
    th2 = th2_0
    om1 = om1_0
    om2 = om2_0

    def alpha1(th1, th2):
        return m2 / (m1 + m2) * l2 / l1 * np.cos(th1 - th2)

    def alpha2(th1, th2):
        return l1 / l2 * np.cos(th1 - th2)

    def f1(th1, th2, om1, om2):
        return -m2 / (m1 + m2) * l2 / l1 * np.sin(th1 - th2) * om2**2 - g / l1 * np.sin(th1)

    def f2(th1, th2, om1, om2):
        return l1 / l2 * np.sin(th1 - th2) * om1**2 - g / l2 * np.sin(th2)

    def g1(th1, th2, om1, om2):
        return (f1(th1, th2, om1, om2) - alpha1(th1, th2) * f2(th1, th2, om1, om2)) / (1 - alpha1(th1, th2) * alpha2(th1, th2))

    def g2(th1, th2, om1, om2):
        return (-alpha2(th1, th2) * f1(th1, th2, om1, om2) + f2(th1, th2, om1, om2)) / (1 - alpha1(th1, th2) * alpha2(th1, th2))

    nt = 100000
    th1s = [th1]
    th2s = [th2]
    om1s = [om1]
    om2s = [om2]

    for t in range(nt):
        th1n = th1 + dt * (om1)
        th2n = th2 + dt * (om2)
        om1n = om1 + dt * g1(th1, th2, om1, om2)
        om2n = om2 + dt * g2(th1, th2, om1, om2)

        th1, th2, om1, om2 = th1n, th2n, om1n, om2n
        th1s.append(th1)
        th2s.append(th2)
        om1s.append(om1)
        om2s.append(om2)
    return th1s, th2s, om1s, om2s


def double_pendulum():
    th1, th2 = dynamicsymbols('theta_1 theta_2')
    t, g, m1, m2, l1, l2, = symbols('t g m_1 m_2 l_1 l_2')

    om1 = diff(th1, t)
    om2 = diff(th2, t)

    # Equations for KE
    #v1 = l * om1
    #v2 = l * (om1 + cos(th1 - th2) * om2)
    T = 1. / 2. * m1 * l1**2 * om1**2 + 1. / 2. * m2 * (l1**2 * om1**2 +
                                                        l2**2 * om2**2 +
                                                        2 * l1 * l2 * cos(th1 - th2) * om1 * om2)

    # Equations for PE.
    y1 = -l1 * cos(th1)
    y2 = - l1 * cos(th1) - l2 * cos(th2)
    V = m1 * g * y1 + m2 * g * y2

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
    if True:
        l1 = 1
        l2 = 2
        th1s, th2s, om1s, om2s = double_pendulum_impl(th2_0=3, l2=l2)
        plot_double_pendulum(l1, l2, th1s, th2s)

    if False:
        eq1, eq2 = double_pendulum()
    if False:
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
