from __future__ import division, print_function
'''Modified from https://gist.github.com/tboggs/8778945'''

'''Functions for drawing contours of Dirichlet distributions.'''

# Original Author: Thomas Boggs
# Modified by Wei-Ting Chiu
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os
import imageio
import itertools

_corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
_AREA = 0.5 * 1 * 0.75**0.5
_triangle = tri.Triangulation(_corners[:, 0], _corners[:, 1])

# For each corner of the triangle, the pair of other corners
_pairs = [_corners[np.roll(range(3), -i)[1:]] for i in range(3)]
# The area of the triangle formed by point xy and another pair or points
tri_area = lambda xy, pair: 0.5 * np.linalg.norm(np.cross(*(pair - xy)))

def xy2bc(xy, tol=1.e-4):
    '''Converts 2D Cartesian coordinates to barycentric.
    Arguments:
        `xy`: A length-2 sequence containing the x and y value.
    '''
    coords = np.array([tri_area(xy, p) for p in _pairs]) / _AREA
    return np.clip(coords, tol, 1.0 - tol)

class Dirichlet(object):
    def __init__(self, alpha):
        '''Creates Dirichlet distribution with parameter `alpha`.'''
        from math import gamma
        from operator import mul
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / \
                     np.multiply.reduce([gamma(a) for a in self._alpha])
    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        from operator import mul
        return self._coef * np.multiply.reduce([xx ** (aa - 1)
                                                for (xx, aa)in zip(x, self._alpha)])
    def sample(self, N):
        '''Generates a random sample of size `N`.'''
        return np.random.dirichlet(self._alpha, N)

def draw_pdf_contours(dist, border=False, nlevels=200, subdiv=8, **kwargs):
    '''Draws pdf contours over an equilateral triangle (2-simplex).
    Arguments:
        `dist`: A distribution instance with a `pdf` method.
        `border` (bool): If True, the simplex border is drawn.
        `nlevels` (int): Number of contours to draw.
        `subdiv` (int): Number of recursive mesh subdivisions to create.
        kwargs: Keyword args passed on to `plt.triplot`.
    '''
    from matplotlib import ticker, cm
    import math

    refiner = tri.UniformTriRefiner(_triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]

    plt.tricontourf(trimesh, pvals, nlevels, cmap='jet', **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')
    plt.colorbar()
    if border is True:
        plt.triplot(_triangle, linewidth=1)

def plot_points(X, color, ms, barycentric=True, border=True, filename = 'tbd', **kwargs):
    '''Plots a set of points in the simplex.
    Arguments:
        `X` (ndarray): A 2xN array (if in Cartesian coords) or 3xN array
                       (if in barycentric coords) of points to plot.
        `barycentric` (bool): Indicates if `X` is in barycentric coords.
        `border` (bool): If True, the simplex border is drawn.
        kwargs: Keyword args passed on to `plt.plot`.
    '''
    if barycentric is True:
        X = X.dot(_corners)
    if X.ndim > 1: 
        plt.plot(X[:, 0], X[:, 1], color+'.', ms=ms, **kwargs)
    else:
        plt.plot(X[0], X[1], color+'.', ms=ms, **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')
    if border is True:
        p = plt.triplot(_triangle, linewidth=1)
    return p


def createGIF(cols, gif_name, max_iter, n_frames = 5, markersize = 10):        
    filenames = []
    for idx in range(int(max_iter/100)):
        if idx % 100 == 0:
            filename = f'images/frame_{idx}.png'
            filenames.append(filename)
            plot_points(cols[0][idx], 'r', markersize)
            plot_points(cols[1][idx], 'g', markersize)
            plot_points(cols[2][idx], 'b', markersize)
            for j in range(n_frames):
                filenames.append(filename)
            plt.savefig(filename)
            plt.close()

    # Build GIF
    print('creating gif')
    with imageio.get_writer(f'{gif_name}.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    print('gif complete')
    print('Removing Images')
    # Remove files
    for filename in set(filenames):
        os.remove(filename)
    print('done...')

def autocorr(x,lags,var):
    n_vectors = len(x)
    nr, nc = len(x[0]), len(x[0][0])
    mean = x.mean(axis = 0)
    xp = np.array([row-mean for row in x])
    corr = np.array([np.correlate(xp[:,r,c],xp[:,r,c],'full') for r, c in itertools.product(range(nr), range(nc))])[:, n_vectors-1:]
    div = np.array([nr-i for i in range(len(lags))])
    acorr = corr.sum()[:len(lags)]/var/div

    return acorr[:len(lags)]

def running_average(x):
    x = np.array(x)
    nr = len(x)
    nc = len(x[0])
    ra = [x[0]]
    for i in range(1, nr):
        ra.append(ra[i-1] + x[i])
    for i in range(1, nr):
        ra[i] /= (i+1)
    return ra

if __name__ == '__main__':
    pass