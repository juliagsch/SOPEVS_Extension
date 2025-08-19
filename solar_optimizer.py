import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy import array
import datetime
import ast
import read_polyshape_3d


def pair_meshes_to_quads(meshes):
    """Pairs adjacent triangular meshes into quadrilaterals with proper vertex deduplication."""
    sorted_keys = sorted(meshes.keys(), key=lambda x: int(x.split('_')[1]))
    quads = {}

    for i in range(0, len(sorted_keys), 2):
        if i + 1 >= len(sorted_keys):
            break

        key1, key2 = sorted_keys[i], sorted_keys[i + 1]
        mesh1, mesh2 = meshes[key1], meshes[key2]

        # Combine coordinates from both meshes
        combined_coords = np.vstack([mesh1['original_coordinates'],
                                     mesh2['original_coordinates']])

        # Handle floating-point precision issues by rounding
        rounded_coords = np.round(combined_coords, decimals=8)

        # Get unique coordinates while preserving original values
        _, unique_indices = np.unique(rounded_coords, axis=0, return_index=True)
        unique_coords = combined_coords[unique_indices].tolist()

        if len(unique_coords) != 4:
            print(f"Pair {key1}-{key2}: Found {len(unique_coords)} unique points. Requires 4.")
            continue

        # Calculate quad properties
        avg_radiance = (mesh1['average_radiance'] + mesh2['average_radiance']) / 2
        centroid = np.mean(unique_coords, axis=0)

        # Order points in quadrilateral sequence
        ordered_coords = _order_quad_points(unique_coords)

        quads[f"Quad_{i // 2 + 1}"] = {
            'coordinates': ordered_coords,
            'average_radiance': avg_radiance,
            'centroid': centroid
        }

    return quads


def _order_quad_points(points):
    """Orders points in a convex quadrilateral in sequential order."""
    # Convert to numpy array for calculations
    pts = np.array(points)

    # Find center point
    center = np.mean(pts, axis=0)

    # Calculate angles from center to sort points radially
    vectors = pts[:, :2] - center[:2]
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])

    # Sort points based on angles
    ordered_indices = np.argsort(angles)

    # Return ordered points (ensure closed polygon)
    ordered = pts[ordered_indices].tolist()
    return ordered + [ordered[0]]  # Close the polygon if needed for visualization


def visualize_quads(quads, output_path):
    """Visualizes quadrilateral meshes in 3D with radiance-based coloring."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Setup colormap
    radiances = [q['average_radiance'] for q in quads.values()]
    norm = plt.Normalize(min(radiances), max(radiances))
    cmap = plt.cm.viridis

    # Plot each quadrilateral
    for quad in quads.values():
        poly = Poly3DCollection([quad['coordinates']],
                                facecolor=cmap(norm(quad['average_radiance'])),
                                edgecolor='k', alpha=0.8)
        ax.add_collection3d(poly)

    # Configure plot
    ax.set(xlabel='X', ylabel='Y', zlabel='Z',
           title='3D Visualization of Quadrilateral Meshes')

    # Add colorbar
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                 ax=ax, label='Average Radiance', shrink=0.5)

    ax.view_init(elev=30, azim=45)
    #plt.show()
    #plt.savefig(output_path, dpi=300, bbox_inches='tight')
    #plt.close()  # Important to free memory


def optimize_panel_placement(quads, num_panels, quad_size, panel_length, panel_width):
    """Optimizes solar panel placement based on radiance values."""
    # Prepare grid structure
    rows = _create_quad_rows(quads)

    # Calculate panel dimensions in quad units
    quads_x = max(1, int(round(panel_length / quad_size)))
    quads_y = max(1, int(round(panel_width / quad_size)))

    # Generate valid panels
    panels = _generate_valid_panels(rows, quads_x, quads_y)

    # Select optimal non-overlapping panels
    selected = _select_optimal_panels(panels, num_panels)

    return _process_panels(selected)


def _create_quad_rows(quads):
    """Organize quads into rows based on centroid coordinates."""
    centroids = [(q['centroid'][0], q['centroid'][1], q)
                 for q in quads.values()]
    sorted_centroids = sorted(centroids, key=lambda c: (-c[1], c[0]))

    # Calculate row grouping tolerance
    y_coords = [c[1] for c in sorted_centroids]
    epsilon = np.max(np.abs(np.diff(y_coords))) * 1.10

    # Group into rows
    rows = []
    current_row = []
    previous_y = None

    for c in sorted_centroids:
        if previous_y is None or abs(c[1] - previous_y) > epsilon:
            if current_row:
                rows.append(sorted(current_row, key=lambda q: q['centroid'][0]))
            current_row = [c[2]]
        else:
            current_row.append(c[2])
        previous_y = c[1]

    if current_row:
        rows.append(sorted(current_row, key=lambda q: q['centroid'][0]))

    return rows


def _generate_valid_panels(rows, quads_x, quads_y):
    """Generate all possible valid panels in the grid."""
    panels = []
    num_rows = len(rows)

    for i in range(num_rows - quads_y + 1):
        for j in range(len(rows[i]) - quads_x + 1):
            valid = True
            covered = []
            total_rad = 0.0

            for dy in range(quads_y):
                row_idx = i + dy
                if row_idx >= num_rows or j + quads_x > len(rows[row_idx]):
                    valid = False
                    break

                for dx in range(quads_x):
                    quad = rows[row_idx][j + dx]
                    total_rad += quad['average_radiance']
                    covered.append(quad)

            if valid:
                panels.append({
                    'total_radiance': total_rad,
                    'quads': covered,
                    'start_row': i,
                    'start_col': j
                })

    return sorted(panels, key=lambda x: -x['total_radiance'])


def _select_optimal_panels(panels, num_panels):
    """Select non-overlapping panels using greedy algorithm."""
    selected = []
    used_quads = set()

    for panel in panels:
        if len(selected) >= num_panels:
            break

        conflict = any(id(q) in used_quads for q in panel['quads'])
        if not conflict:
            selected.append(panel)
            used_quads.update(id(q) for q in panel['quads'])

    return selected


def _process_panels(panels):
    """Process selected panels into final output format."""
    processed = []
    total_rad = 0.0

    for panel in panels:
        all_points = [pt for q in panel['quads'] for pt in q['coordinates']]
        min_x, max_x = np.min([p[0] for p in all_points]), np.max([p[0] for p in all_points])
        min_y, max_y = np.min([p[1] for p in all_points]), np.max([p[1] for p in all_points])
        avg_z = np.mean([p[2] for p in all_points])

        processed.append({
            'original_coordinates': [q['coordinates'] for q in panel['quads']],
            'new_coordinates': [
                [min_x, max_y, avg_z],
                [max_x, max_y, avg_z],
                [max_x, min_y, avg_z],
                [min_x, min_y, avg_z]
            ],
            'total_radiance': panel['total_radiance'],
            'position': (panel['start_row'], panel['start_col'])
        })
        total_rad += panel['total_radiance']

    return {
        'panels': processed,
        'total_radiance': total_rad
    }


def visualize_quads_and_panels(quads, panels_data, output_path):
    """Combined visualization of quads and solar panels."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot base quads
    radiances = [q['average_radiance'] for q in quads.values()]
    norm = plt.Normalize(min(radiances), max(radiances))
    cmap = plt.cm.viridis

    for quad in quads.values():
        poly = Poly3DCollection([quad['coordinates']],
                                facecolor=cmap(norm(quad['average_radiance'])),
                                edgecolor='gray', alpha=0.4)
        ax.add_collection3d(poly)

    # Plot panels
    colors = ['grey']
    for i, panel in enumerate(panels_data['panels']):
        for quad in panel['original_coordinates']:
            poly = Poly3DCollection([quad],
                                    facecolor=colors[i % len(colors)],
                                    edgecolor='black', alpha=0.6)
            ax.add_collection3d(poly)

    ax.set(xlabel='Longitude', ylabel='Latitude', zlabel='Elevation',
           title='Solar Panel Placement Visualization')

    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                 ax=ax, label='Average Radiance', shrink=0.5)

                                                                              # for single_segment, 20 -45
                                                                              # for test2, smaller face 20 0
                                                                              # for test2, larger face 20, -90
                                                                              # new complex building, 20, 90

    #plt.show()
    # read_polyshape_3d.set_axes_equal(ax)
    # plt.show()
    ax.view_init(elev=20, azim=135)                                           # change the viewing angle here

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()  # Important to free memory





def run_optimization(meshes, panel_length, panel_width,num_panels, output_path):
    """Main function to be called from C++ with panel dimensions"""
    # Hardcoded mesh data (truncated for brevity)


    quads = pair_meshes_to_quads(meshes)
    result = optimize_panel_placement(
        quads,
        num_panels,
        quad_size=1,
        panel_length=panel_length,
        panel_width=panel_width
    )
    #visualize_quads_and_panels(quads, result)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    quads_path = os.path.join(output_path, f'quads_{timestamp}.png')
    panels_path = os.path.join(output_path, f'panels_{timestamp}.png')
    
    #visualize_quads(quads, quads_path)
    visualize_quads_and_panels(quads, result, panels_path)

    return {
        'total_radiance': result['total_radiance'],
        'num_panels': len(result['panels'])
    }



def read_and_remove_results(path):
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(path, "comprehensive_results.txt")

    try:
        with open(filename, 'r') as f:
            content = f.read().strip()
        
        # Use JSON instead of ast.literal_eval
        import json
        data = json.loads(content)
        
        # os.remove(filename)
        return data
    except Exception as e:
        print(f"Error reading results: {e}")
        return None
    
    

def main():
    import sys
    if len(sys.argv) != 5:
        print("Usage: solar_optimizer.py <num_panels> <panel_length> <panel_width> <output_path>")
        sys.exit(1)
    
    image_save = "."
    comprehensive_results = read_and_remove_results(sys.argv[4])

    optimization_result = run_optimization(
        comprehensive_results, 
        panel_length= int(sys.argv[2]),
        panel_width= int(sys.argv[3]),
        num_panels= int(sys.argv[1]),
        output_path=image_save
    )

    return True

if __name__ == '__main__':
    main()
