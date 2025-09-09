from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

elev, azim = 25, -90
exclude_axis_scale = True

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale.'''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    # Calculate midpoints
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    # Set new limits with the same range
    ax.set_xlim3d(x_middle - max_range / 2, x_middle + max_range / 2)
    ax.set_ylim3d(y_middle - max_range / 2, y_middle + max_range / 2)
    ax.set_zlim3d(z_middle - max_range / 2, z_middle + max_range / 2)

    if exclude_axis_scale:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])


def plot_building(verts, faces):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each face
    for face in faces:
        # Get vertices for this face (convert to 0-based index)
        polygon = [verts[i] for i in face]
        # Create polygonal patch
        poly = Poly3DCollection([polygon], alpha=0.5, edgecolor='k', linewidths=1)
        poly.set_facecolor(np.random.rand(3, ))
        ax.add_collection3d(poly)

    min_vals = verts.min(axis=0)
    max_vals = verts.max(axis=0)
    ax.set_xlim(min_vals[0], max_vals[0])
    ax.set_ylim(min_vals[1], max_vals[1])
    ax.set_zlim(min_vals[2], max_vals[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)

    ax.view_init(elev=elev, azim=azim)
    plt.show()


def plot_rooftops(verts, roof_faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot all vertices and rooftop faces
    for face in roof_faces:
        poly = Poly3DCollection([verts[face]], alpha=0.5, edgecolor='k', linewidths=1)
        ax.add_collection3d(poly)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)

    ax.view_init(elev=elev, azim=azim)
    plt.show()


def plot_rooftops_with_mesh_points(roof_v, roof_faces, mesh_objects):
    """
    Plots the rooftop structure along with the generated mesh grid on top.

    Arguments can be added/modified if necessary
    NO Args:
        verts (np.ndarray): Nx3 array of vertex coordinates.
        roof_faces (List[List[int]]): List of faces, where each face is a list of vertex indices.
        mesh_objects (List[List[List[float]]]): List of mesh squares for each face,
                                                where each square is a list of 3D points.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot rooftop faces
    for face in roof_faces:
        poly = roof_v[face]
        poly = Poly3DCollection([poly], alpha=0.5, edgecolor='k', linewidths=1)
        poly.set_facecolor(np.random.rand(3, ))
        ax.add_collection3d(poly)

    # Plot mesh grid points
    for square in mesh_objects:
        square_points = np.array(square)
        ax.scatter(square_points[:, 0], square_points[:, 1], square_points[:, 2], c='red', s=2, alpha=0.7)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)

    ax.view_init(elev=elev, azim=azim)
    ax.set_title("Rooftop with Mesh Points")
    plt.legend(loc='upper left', markerscale=2, fontsize=8)
    plt.show()


def plot_rooftops_with_mesh_grid(roof_v, roof_faces, mesh_objects, surroundings_v, surroundings_f):
    """
    Plots the rooftop structure along with the generated mesh grid as quadrilaterals.

    NO Args:
        verts (np.ndarray): Nx3 array of vertex coordinates.
        roof_faces (List[List[int]]): List of faces, where each face is a list of vertex indices.
        mesh_objects (List[List[List[float]]]): List of mesh squares for each face,
                                                where each square is a list of 3D points.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot all vertices and rooftop faces
    for face in roof_faces:
        poly = roof_v[face]
        poly = Poly3DCollection([poly], alpha=0.5, edgecolor='k', linewidths=1)
        poly.set_facecolor(np.random.rand(3, ))
        ax.add_collection3d(poly)

    # Plot the mesh grid as quadrilaterals
    for square in mesh_objects:
        square_points = np.array(square)
        poly = Poly3DCollection([square_points], alpha=0.8, edgecolor='blue', linewidths=0.5)
        poly.set_facecolor('blue')
        ax.add_collection3d(poly)

    for face in surroundings_f:
        poly = Poly3DCollection([surroundings_v[face]], alpha=0.4, edgecolor='k', linewidths=1)
        poly.set_facecolor((0.2,0.2,0.2))
        ax.add_collection3d(poly)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    ax.view_init(elev=elev, azim=azim)

    ax.set_title("Rooftop with Mesh Grid")
    plt.legend(loc='upper left', markerscale=2, fontsize=8)
    plt.show()


def plot_surroundings(surroundings_v, surroundings_f):
    """
    Plots the occlusion mesh.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for face in surroundings_f:
        poly = Poly3DCollection([surroundings_v[face]], alpha=0.3, edgecolor='k', linewidths=1)
        poly.set_facecolor((0.4,0.4,0.4))
        ax.add_collection3d(poly)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    ax.view_init(elev=elev, azim=azim)

    ax.set_title("Occlusion Mesh")
    plt.legend(loc='upper left', markerscale=2, fontsize=8)
    plt.show()


def plot_building_with_mesh_grid(roof_v, roof_f, mesh_objects):
    """
    Plots the 3D building with the generated mesh grid on top.

    NO Args:
        verts (np.ndarray): Nx3 array of vertex coordinates.
        faces (List[List[int]]): List of faces, where each face is a list of vertex indices.
        mesh_objects (List[List[List[float]]]): List of mesh squares for each face,
                                                where each square is a list of 3D points.
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each face of the building
    for face in roof_f:
        polygon = [roof_v[i] for i in face]
        poly = Poly3DCollection([polygon], alpha=0.5, edgecolor='k', linewidths=1)
        poly.set_facecolor(np.random.rand(3, ))
        ax.add_collection3d(poly)

    # Plot the mesh grid as quadrilaterals
    for square in mesh_objects:
        square_points = np.array(square)
        poly = Poly3DCollection([square_points], alpha=0.8, edgecolor='blue', linewidths=0.5)
        poly.set_facecolor('blue')
        ax.add_collection3d(poly)

    min_vals = roof_v.min(axis=0)
    max_vals = roof_v.max(axis=0)
    ax.set_xlim(min_vals[0], max_vals[0])
    ax.set_ylim(min_vals[1], max_vals[1])
    ax.set_zlim(min_vals[2], max_vals[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)

    ax.view_init(elev=elev, azim=azim)
    plt.title('Building Model with Mesh Grid')
    plt.show()


def visualize_quads_and_panels(quads, panel_clusters, surroundings=None, kwh_per_panel=0.4):
    """Combined visualization of quads and solar panels."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    norm = plt.Normalize(vmin=0, vmax=1200)  # fixed scale for better comparison between buildings
    cmap = plt.cm.viridis

    # Plot base quads
    for quad in quads.values():
        poly = Poly3DCollection([quad['original_coordinates']],
                                facecolor=cmap(norm(round(quad['average_radiance']*8760/1000,1))),
                                edgecolor='gray', alpha=0.9)
        ax.add_collection3d(poly)

    # Plot panels
    total_production = 0
    for i, panel in enumerate(panel_clusters):
        for quad_name in panel.quads:
            poly = Poly3DCollection([quads[quad_name]['original_coordinates']],
                                    facecolor='red',
                                    edgecolor='black', alpha=1)
            ax.add_collection3d(poly)

        total_production += (panel.total_production*kwh_per_panel)

    if surroundings is not None:
        surroundings_V, surroundings_F = surroundings
        for i, face in enumerate(surroundings_F):
            poly = Poly3DCollection([surroundings_V[face]],
                                    facecolor='black',
                                    edgecolor='black', alpha=0.3)
            ax.add_collection3d(poly)

    ax.set(xlabel='Longitude', ylabel='Latitude', zlabel='Elevation',
           title='Solar Panel Placement Visualization')

    # Colorbar
    print("Total yearly production of panel layout", total_production)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                        ax=ax, label='Yearly Production per kW', shrink=0.5)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f kWh"))
    set_axes_equal(ax)

    ax.view_init(elev=elev, azim=azim)

    plt.show()
    plt.close()

def visualize_panels(quads, panel_clusters, surroundings=None, kwh_per_panel=0.4):
    """Combined visualization of quads and solar panels."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot panels
    total_production = 0
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i, panel in enumerate(panel_clusters):
        for quad_name in panel.quads:
            poly = Poly3DCollection([quads[quad_name]['original_coordinates']],
                                    facecolor='black',
                                    edgecolor='black', alpha=0.8)
            ax.add_collection3d(poly)
        print(panel.total_production*kwh_per_panel)
        total_production += (panel.total_production*kwh_per_panel)

    if surroundings is not None:
        surroundings_V, surroundings_F = surroundings
        for i, face in enumerate(surroundings_F):
            poly = Poly3DCollection([surroundings_V[face]], alpha=0.4, edgecolor='gray', facecolor=colors[i % len(colors)], linewidths=1)
            ax.add_collection3d(poly)

    ax.set(xlabel='Longitude', ylabel='Latitude', zlabel='Elevation',
           title='Solar Panel Placement Visualization')

    print("Total yearly production of panel layout", total_production)
    set_axes_equal(ax)
    ax.view_init(elev=elev, azim=azim)
    plt.show()
    plt.close()


def visualize_ray_trace(origin, solar_dir, occlusion_triangles, intersection=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot occlusion triangles
    for tri in occlusion_triangles:
        poly = Poly3DCollection([tri], alpha=0.3, facecolor="gray", edgecolor="k")
        ax.add_collection3d(poly)

    # Plot ray origin
    ax.scatter(*origin, color="red", s=20, label="Origin")

    # Plot solar ray (extend it for visualization)
    start = origin + solar_dir * 0.25  # make the arrow long enough
    ax.quiver(*start, *solar_dir, length=10, color="orange", label="Solar Ray")

    # Plot intersection if any
    if intersection is not None:
        ax.scatter(*intersection, color="blue", s=80, label="Intersection")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()