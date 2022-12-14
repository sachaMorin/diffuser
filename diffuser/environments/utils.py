import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import matplotlib.tri as mtri

from scipy.sparse.csgraph import shortest_path, connected_components


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().

    Credits to Karlo from https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def surface_plot(x, y=None, s=1, fig=None, ax=None, tilt=30, rotation=-80, edgecolor='k', c=None, cmap="rainbow"):
    """3D plot of the data
    Args:
        x(ndarray): Points.
        y(ndarray): Labels for coloring.
        s(int): Marker size.
        tilt(int): Inclination towards observer.
        rotation(rotation): Rotation angle.
        edgecolor(str): Edge color.
    """
    if y is None:
        y = np.ones(x.shape[0])

    if fig is None:
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.view_init(tilt, rotation)

    if c is None:
        c = np.arange(x.shape[0])

    x[:, 2] *= 1.15
    ax.scatter(*x.T, cmap=cmap, c=c, s=s)
    set_axes_equal(ax)
    return fig, ax


def triu_plot(x, coords, s=20, tilt=30, rotation=-80, edgecolor='k'):
    """3D plot of the data
    Args:
        x(ndarray): 3D points.
        coords(ndarray): Coordinates for triangulation
        s(int): Marker size.
        tilt(int): Inclination towards observer.
        rotation(rotation): Rotation angle.
        edgecolor(str): Edge color.
    """
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.view_init(tilt, rotation)

    # Triangulate parameter space to determine the triangles
    tri = mtri.Triangulation(*coords.T)
    x[:, 2] *= 1.15
    ax.plot_trisurf(*x.T, color='white', alpha=0.1, triangles=tri.triangles)
    set_axes_equal(ax)

    return fig, ax
