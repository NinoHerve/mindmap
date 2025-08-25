import numpy as np
import pyvista as pv
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt



def generate_uniform_sphere_points(n_points: int) -> np.ndarray:
    """
    Generates n_points uniformly distributed on the surface of a unit sphere.

    Args:
        n_points (int): The number of points to generate.

    Returns:
        np.ndarray: An array of shape (n_points, 3) where each row is a 
                    (x, y, z) coordinate of a point on the unit sphere.
    """
    # Generate n_points uniform random numbers for the polar angle (phi)
    # in the range [0, 2*pi).
    phi = 2 * np.pi * np.random.rand(n_points)

    # Generate n_points uniform random numbers for the polar angle (theta)
    # The cosine of theta is uniformly distributed in the range [-1, 1].
    # This is a key step to ensure a uniform distribution.
    cos_theta = 2 * np.random.rand(n_points) - 1
    
    # Calculate the sine of theta
    sin_theta = np.sqrt(1 - cos_theta**2)

    # Convert spherical coordinates to Cartesian coordinates
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = cos_theta

    # Stack the coordinates into a single array
    points = np.stack([x, y, z], axis=1)

    return points


def euclidean_distance_matrix(points: np.ndarray) -> np.ndarray:
    """
    Measures the Euclidean distance between all pairs of points.

    Args:
        points (np.ndarray): An array of shape (n_points, n_dimensions).

    Returns:
        np.ndarray: A symmetric distance matrix of shape (n_points, n_points).
                    The element at (i, j) is the Euclidean distance between
                    point i and point j.
    """
    # Use pdist to compute the condensed distance vector
    condensed_distances = pdist(points, 'euclidean')

    # Use squareform to convert the condensed vector to a full matrix
    distance_matrix = squareform(condensed_distances)

    return distance_matrix

def make_kernel(eta, mu1, var1, mu2, var2):
    kernel = lambda x: eta * (mu1*np.exp(-var1*x*x) + mu2*np.exp(-var2*x*x))
    return kernel


def plot_graph(points, adjacency):
    """
    Parameters
    ----------
    points: np.array (N,3) 
    lines: np.array (N,2)
    """

    P = pv.PolyData(points)
    A1 = adjacency > 0
    A2 = adjacency < 0

    lines1 = []
    lines2 = []
    for i in range(len(A1)):
        for j in range(i, len(A1)):
            if A1[i,j]:
                lines1 += [points[i], points[j]]
            if A2[i,j]:
                lines2 += [points[i], points[j]]
    lines1 = np.array(lines1)
    lines2 = np.array(lines2)

    p = pv.Plotter()
    p.add_mesh(P)
    if len(lines1) > 0:
        p.add_lines(lines1, color="green", width=0.05)
    if len(lines2) > 0:
        p.add_lines(lines2, color="red", width=0.01)

    return p

def process_data(time, data, range_):
    # process data
    data_ = (data[:,1,:,:] - data[:,2,:,:]).squeeze().astype(np.float32)
    data_ = data_[range_[0]:range_[1], :]
    time_ = time[range_[0]:range_[1]]

    #data_ = data_ / np.max(np.abs(data_))

    return time_, data_

def plot_graph_activity(conn, time, data, range_ = [2000, -1], show_max=False):

    time, data = process_data(time, data, range_ = range_)
    clim = [-np.max(np.abs(data)), np.max(np.abs(data))]

    p = pv.Plotter()

    def create_mesh(t):
        points = pv.PolyData(conn.centres)
        points["activity"] = data[int((t-time[0])/2**-1),:]
        p.add_mesh(points, point_size=20,render_points_as_spheres=True, clim=clim)

    p.add_slider_widget(create_mesh, [time[0], time[-1]], title="time [ms]")

    if show_max:
        # nodes with biggest spikes
        maxis = np.max(data, axis=0)
        indices = np.argsort(maxis)[-1]

        p.add_mesh(pv.PolyData(conn.centres[indices]), point_size=50, color="red")
        
    p.show()

def plot_activity(time, data, node=None, start_idx=2000):

    fig, axs = plt.subplots(nrows=10, ncols=1, figsize=(20,15), sharex = True)

    time, data = process_data(time, data, range_=[start_idx, -1])
    
    if node is None:
        maxis = np.max(data, axis=0)
        indices = np.argsort(maxis)[-10:]
    else:
        indices = np.arange(node-5, node+5)

    for i,j in enumerate(indices):
        axs[i].plot(time, data[:, j])
        axs[i].set_title(f"Node {j}")
    fig.show()