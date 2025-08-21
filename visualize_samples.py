import matplotlib.pyplot as plt
import numpy as np
import os 

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_subdirectories(path="./samples/"):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def read_polyshape(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    faces, verts = [], []
    
    if '.obj' in filename:
        for line in lines:
            parts = line.split(' ')
            if parts[0] == 'v':
                verts.append([float(parts[1]), float(parts[2]),float(parts[3])])
            elif parts[0] == 'f':
                face = []
                for v in parts[1:]:
                    face.append(int(v)-1)
                faces.append(face)
                
        verts = np.array(verts)

        for i in range(len(verts)):
            verts[i,:] = verts[i,0], verts[i,1], verts[i,2]
    
    return verts, faces


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

    ax.view_init(elev=60, azim=-30)
    plt.show()


def plot_building_and_surroundings(building_v, building_f, surroundings_v, surroundings_f):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each building face
    for face in building_f:
        polygon = [building_v[i] for i in face]
        poly = Poly3DCollection([polygon], alpha=0.5, edgecolor='k', linewidths=1)
        poly.set_facecolor(np.random.rand(3, ))
        ax.add_collection3d(poly)
    
    # Plot each surroundings face
    for face in surroundings_f:
        polygon = [surroundings_v[i] for i in face]
        poly = Poly3DCollection([polygon], alpha=0.2, edgecolor='k', linewidths=1)
        poly.set_facecolor((0.3,0.3,0.3))
        ax.add_collection3d(poly)

    min_vals = surroundings_v.min(axis=0)
    max_vals = surroundings_v.max(axis=0)
    ax.set_xlim(min_vals[0], max_vals[0])
    ax.set_ylim(min_vals[1], max_vals[1])
    ax.set_zlim(min_vals[2], max_vals[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)

    ax.view_init(elev=60, azim=-30)
    plt.show()


def plot_elevation_image(surroundings2D_path, mask_path):
    z_grid = np.genfromtxt(surroundings2D_path, delimiter=",")
    mask = np.genfromtxt(mask_path, delimiter=",")

    # Handle NaN
    invalid_mask = np.isnan(z_grid)
    valid_values = z_grid[~invalid_mask]

    # Normalize to [0, 1]
    z_min_norm = valid_values.min()
    z_max_norm = valid_values.max()
    normalized = (z_grid - z_min_norm) / (z_max_norm - z_min_norm)
    normalized[invalid_mask] = 1.0

    # Invert the gradient
    inverted = 1.0 - normalized

    inverted = inverted[::-1, :]
    mask = mask[::-1, :]

    # Create an RGB image
    h, w = inverted.shape
    rgb_img = np.zeros((h, w, 3))

    rgb_img[:, :, 0] = inverted * (mask == 1) + inverted * (mask == 0)
    rgb_img[:, :, 1] = inverted * (mask == 0)
    rgb_img[:, :, 2] = inverted * (mask == 0)

    plt.imshow(rgb_img, vmin=0, vmax=1)
    plt.axis("off")
    plt.show()

samples = get_subdirectories()
samples = ["2615350_1234696_2615364_1234710", "2613235_1235070_2613243_1235083", "2613096_1233451_2613111_1233466", "2615189_1234471_2615197_1234479",
"2615150_1233155_2615160_1233165"]
for sample in samples:
    print(sample)
    vertices, faces = read_polyshape(os.path.join("./samples", sample, "building.obj"))
    vertices_surroundings, faces_surroundings = read_polyshape(os.path.join("./samples", sample, "surroundings3D.obj"))

    plot_building(vertices, faces)
    plot_building_and_surroundings(vertices, faces, vertices_surroundings, faces_surroundings)
    # plot_elevation_image(os.path.join("./samples", sample, "surroundings2D.csv"), os.path.join("./samples", sample, "mask.csv"))

"not a lot of shading:"
"2615350_1234696_2615364_1234710"
""
""
"shading"
"2613235_1235070_2613243_1235083"
""
""
"complex no shading:"
"2613096_1233451_2613111_1233466"
""
"small no shading: "
"2615189_1234471_2615197_1234479"
"2615150_1233155_2615160_1233165"