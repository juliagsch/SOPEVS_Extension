from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

elev, azim = 25, -90

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


def plot_solar_access(shaded, unshaded, solar_azimuth, solar_zenith):
    """
    Visualizes 3D solar access analysis by plotting shaded and unshaded areas.

    Parameters:
    - shaded (list of arrays): List of polygons representing shaded areas.
    - unshaded (list of arrays): List of polygons representing unshaded areas.
    - solar_azimuth (float): Solar azimuth angle in degrees for the title.
    - solar_zenith (float): Solar zenith angle in degrees for the title.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot UNSHADED areas (green)
    if unshaded:
        unshaded_collection = Poly3DCollection(
            unshaded,
            facecolors='#00ff00',
            edgecolors='#003300',
            linewidths=0.3,
            alpha=0.9,
            zorder=2
        )
        ax.add_collection3d(unshaded_collection)

    # Plot SHADED areas (red) on top
    if shaded:
        shaded_collection = Poly3DCollection(
            shaded,
            facecolors='#ff3300',
            edgecolors='#660000',
            linewidths=0.3,
            alpha=0.8,
            zorder=3
        )
        ax.add_collection3d(shaded_collection)

    # Set axes limits based on combined points
    if shaded or unshaded:
        all_points = np.concatenate(shaded + unshaded) if shaded else np.concatenate(unshaded)
        min_vals = np.min(all_points, axis=0)
        max_vals = np.max(all_points, axis=0)

        padding = 0.1 * (max_vals - min_vals)
        ax.set_xlim(min_vals[0] - padding[0], max_vals[0] + padding[0])
        ax.set_ylim(min_vals[1] - padding[1], max_vals[1] + padding[1])
        ax.set_zlim(min_vals[2] - padding[2], max_vals[2] + padding[2])

    # Configure view and labels
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X Axis', fontsize=10, labelpad=10)
    ax.set_ylabel('Y Axis', fontsize=10, labelpad=10)
    ax.set_zlabel('Elevation', fontsize=10, labelpad=10)
    ax.set_title(
        f'Solar Access Map\nAzimuth: {solar_azimuth}°, Zenith: {solar_zenith}°',
        fontsize=12, pad=15
    )

    # Add legend
    legend_elements = [
        plt.matplotlib.patches.Patch(facecolor='#00ff00', alpha=0.9, label='Direct Sunlight'),
        plt.matplotlib.patches.Patch(facecolor='#ff3300', alpha=0.8, label='Shaded Areas')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Optimize 3D rendering
    plt.tight_layout()
    ax.xaxis.set_pane_color((0.95, 0.95, 0.95))
    ax.yaxis.set_pane_color((0.95, 0.95, 0.95))
    ax.zaxis.set_pane_color((0.97, 0.97, 0.97))
    ax.grid(False)

    plt.show()


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
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c='grey', alpha=0.3)
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
    ax.scatter(roof_v[:, 0], roof_v[:, 1], roof_v[:, 2], c='grey', alpha=0.3)

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
    ax.scatter(roof_v[:, 0], roof_v[:, 1], roof_v[:, 2], c='grey', alpha=0.3, label='Vertices')
    for face in roof_faces:
        poly = roof_v[face]
        poly = Poly3DCollection([poly], alpha=0.5, edgecolor='k', linewidths=1)
        poly.set_facecolor(np.random.rand(3, ))
        ax.add_collection3d(poly)

    # Plot the mesh grid as quadrilaterals
    for square in mesh_objects:
        square_points = np.array(square)
        x = square_points[:, 0]
        y = square_points[:, 1]
        z = square_points[:, 2]
        ax.plot_trisurf(x, y, z, color='blue', alpha=0.6, edgecolor='black')

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


def visualize_quads_and_panels(quads, panels_data, surroundings=None):
    """Combined visualization of quads and solar panels."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    radiances = [q['average_radiance'] for q in quads.values()]

    norm = plt.Normalize(vmin=min(radiances), vmax=max(radiances))
    cmap = plt.cm.viridis

    # Plot base quads
    for quad in quads.values():
        poly = Poly3DCollection([quad['original_coordinates']],
                                facecolor=cmap(norm(round(quad['average_radiance'],1))),
                                edgecolor='gray', alpha=0.9)
        ax.add_collection3d(poly)

    # Plot panels
    colors = ['red']
    for i, panel in enumerate(panels_data['panels']):
        for quad in panel['original_coordinates']:
            poly = Poly3DCollection([quad],
                                    facecolor=colors[i % len(colors)],
                                    edgecolor='black', alpha=0.8)
            ax.add_collection3d(poly)

    if surroundings is not None:
        surroundings_V, surroundings_F = surroundings
        for face in surroundings_F:
            poly = Poly3DCollection([surroundings_V[face]], alpha=0.3, edgecolor='gray', linewidths=1)
            ax.add_collection3d(poly)

    ax.set(xlabel='Longitude', ylabel='Latitude', zlabel='Elevation',
           title='Solar Panel Placement Visualization')

    # Colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                        ax=ax, label='Average Radiance', shrink=0.5)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
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