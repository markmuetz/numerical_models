from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pylab as plt
from matplotlib import cm


def main():
    nx = 100
    nt = 1000
    dt = 0.001
    K = 0.001

    x = np.linspace(0, 1, nx + 1)
    y = np.linspace(0, 1, nx + 1)
    X, Y = np.meshgrid(x, y)
    dx = x[1] - x[0]
    ka = K * dt / dx**2

    phi = np.zeros_like(X)
    phi[(X - 0.5)**2 + (Y - 0.5)**2 < 0.1] = 1
    times = np.linspace(0, nt*nx, nt + 1)

    for i, t in enumerate(times):
        diff = (ka * (np.roll(phi, -1, axis=0) - 2 * phi + np.roll(phi, 1, axis=0)) +
                ka * (np.roll(phi, -1, axis=1) - 2 * phi + np.roll(phi, 1, axis=1)))
        print(diff.max())
        phi = phi + diff
        if i % 100 == 0:
            fig = plt.figure(1)
            plt.clf()
            ax = fig.gca(projection='3d')
            surf = ax.plot_surface(X, Y, phi, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
            plt.pause(0.01)
    return surf


if __name__ == '__main__':
    surf = main()