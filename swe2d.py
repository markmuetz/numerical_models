import numpy as np
import pylab as plt

DEBUG = False


def swe_lin_arakawaC(u, v, h, dt, dx, dy, H, g, f):
    vu = (v +
          np.roll(v, 1, axis=0) +
          np.roll(v, 1, axis=1) +
          np.roll(np.roll(v, 1, axis=0), 1, axis=1)) / 4
    uv = (u +
          np.roll(u, 1, axis=0) +
          np.roll(u, 1, axis=1) +
          np.roll(np.roll(u, 1, axis=0), 1, axis=1)) / 4

    un = u + (f * vu - g / dx * (h - np.roll(h, 1, axis=0))) * dt
    vn = v - (f * uv + g / dy * (h - np.roll(h, 1, axis=1))) * dt
    hn = h - (H * (((np.roll(u, -1, axis=0) - u) / dx) + (np.roll(v, -1, axis=1) - v) / dy)) * dt
    if DEBUG:
        print('un.max(): {}'.format(un.max()))
        print('vn.max(): {}'.format(vn.max()))
        print('hn.max(): {}'.format(hn.max()))

        print('un diff: {}'.format(un - u))
        print('vn diff: {}'.format(vn - u))
        print('hn diff: {}'.format(hn - u))

    return un, vn, hn


def main(icfn=None, srcfn=None, barrierfn=None):
    nx = 300
    ny = 100
    Lx = 300
    Ly = 100

    dt = 0.0001
    nt = 100000

    H = 100
    g = 9.81
    f = 0

    xh = np.linspace(0, Lx, nx + 1)
    yh = np.linspace(0, Ly, ny + 1)
    dx = xh[1] - xh[0]
    dy = yh[1] - yh[0]
    X, Y = np.meshgrid(xh, yh)

    u = np.zeros((ny + 1, nx + 1))
    v = np.zeros((ny + 1, nx + 1))
    h = np.zeros((ny + 1, nx + 1))

    if icfn:
        icfn(u, v, h, X, Y)

    times = np.linspace(0, nt * dt, nt + 1)

    for i, t in enumerate(times):
        if srcfn:
            srcfn(u, v, h, X, Y, t)
        u, v, h = swe_lin_arakawaC(u, v, h, dt, dx, dy, H, g, f)
        if barrierfn:
            barrierfn(u, v, h, X, Y, t)

        if i % 1000 == 0:
            plt.clf()
            plt.imshow(h, vmin=-1, vmax=1)
            plt.pause(0.01)

if __name__ == '__main__':
    def initial_circles(u, v, h, X, Y):
        h[((X - 25)**2 + (Y - 25)**2 < 10*2) | ((X - 25)**2 + (Y - 75)**2 < 10*2)] = 10.

    def circle_source(u, v, h, X, Y, t):
        h[((X - 25)**2 + (Y - 25)**2 < 10*2) | ((X - 25)**2 + (Y - 75)**2 < 10*2)] = np.cos(t * 20)

    def line_source(u, v, h, X, Y, t):
        h[(X > 25) & (X < 30)] = np.cos(t * 20)

    def single_slit(u, v, h, X, Y, t):
        u[(X > 40) & (X < 45) & ((Y < 40) | (Y > 60))] = 0
        v[(X > 40) & (X < 45) & ((Y < 40) | (Y > 60))] = 0
        h[(X > 40) & (X < 45) & ((Y < 40) | (Y > 60))] = 0

    def double_slit(u, v, h, X, Y, t):
        u[(X > 40) & (X < 45) & ((Y < 25) | ((Y > 35) & (Y < 65)) | (Y > 75))] = 0
        v[(X > 40) & (X < 45) & ((Y < 25) | ((Y > 35) & (Y < 65)) | (Y > 75))] = 0
        h[(X > 40) & (X < 45) & ((Y < 25) | ((Y > 35) & (Y < 65)) | (Y > 75))] = 0

    #main(initial_circles)
    #main(srcfn=line_source, barrierfn=double_slit)
    main(srcfn=circle_source)
