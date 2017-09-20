"""Demonstration of methods of modelling simple pendulum."""
import numpy as np
import pylab as plt

import physical_consts as pc


def run_pendulum(th0, g, r, dt=0.001, nt=10000, method='ct'):
    """Run pendulum with given numerical method.

    :param th0: Initial value of theta.
    :param r: Accn due to gravity.
    :param r: Length of pendulum.
    :param dt: delta time.
    :param nt: Number of iterations.
    :param method: 'ct': centred time, 'two_step': two step.
    :return: (times, thetas)
    """
    th_old = th0
    times = np.linspace(0, dt * nt, nt)
    if method == 'ct':
        th_curr = th0
        thetas = [th_old, th_curr]
        omegas = [0, 0]
        for t in times[2:]:
            th_new = 2 * th_curr - th_old - dt**2 * g / r * np.sin(th_curr)
            omegas.append((th_new - th_old) / (2 * dt))
            th_old = th_curr
            th_curr = th_new
            thetas.append(th_new)

    elif method == 'two_step':
        omega_old = 0
        omega_curr = 0
        thetas = [th_old]
        omegas = [omega_old]
        for t in times[1:]:
            th_curr = th_old + dt * omega_curr
            omega_curr = omega_old - dt * g / r * np.sin(th_curr)

            omega_old = omega_curr
            th_old = th_curr

            thetas.append(th_curr)
            omegas.append(omega_curr)

    return times, np.array(thetas), np.array(omegas)


def pe(m, g, l, theta):
    return m * g * l * (1 - np.cos(theta))


def ke(m, g, l, omega):
    return 0.5 * m * (l * omega)**2


def plot_thetas(th0, times, thetas, g, r, method):
    """Plot theta against time for method.

    :param th0: Initial value of theta.
    :param times: Times of simulation.
    :param thetas: Values of theta.
    :param g: Accn due to gravity.
    :param r: Length of pendulum.
    :param method: Name of method.
    :return:
    """
    plt.figure()
    plt.title(method)
    plt.plot(times, thetas)
    plt.plot(times, th0 * np.cos(times * np.sqrt(g / r)))
    plt.show()

def plot_energy(th0, times, thetas, omegas, m, g, r, method):
    plt.figure()

    plt.title(method)
    pes = pe(m, g, r, thetas)
    kes = ke(m, g, r, omegas)
    #plt.plot(times, pes)
    #plt.plot(times, kes)
    plt.plot(times, pes + kes)
    plt.show()


def main():
    """Entry point."""
    g = pc.g
    r = 1
    th0 = np.pi / 2
    m = 1.

    res = {}
    for method in 'ct', 'two_step':
        times, thetas, omegas = run_pendulum(th0, g, r, method=method)
        res[method] = (times, thetas, omegas)
        # plot_thetas(th0, times, thetas, g, r, method)
        plot_energy(th0, times, thetas, omegas, m, g, r, method)
    return res


if __name__ == '__main__':
    res = main()
