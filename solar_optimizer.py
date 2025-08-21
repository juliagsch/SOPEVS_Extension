import datetime
import numpy as np
import os
import visualize


def optimize_panel_placement(quads, num_panels, quad_size, panel_length, panel_width):
    """Optimizes solar panel placement based on radiance values."""
    # Calculate panel dimensions in quad units
    quads_x = max(1, int(round(panel_length / quad_size)))
    quads_y = max(1, int(round(panel_width / quad_size)))

    # Prepare grid structure
    rows = create_quad_rows(quads)
    panels = generate_valid_panels(rows, quads_x, quads_y)
    selected = select_optimal_panels(panels, num_panels)

    return process_panels(selected)


def create_quad_rows(quads):
    """Organize quads into rows based on centroid coordinates."""
    centroids = [(q['centroid'][0], q['centroid'][1], q) for q in quads.values()]
    sorted_centroids = sorted(centroids, key=lambda c: (-c[1], c[0]))

    # Calculate row grouping tolerance
    y_coords = [c[1] for c in sorted_centroids]
    epsilon = np.max(np.abs(np.diff(y_coords))) * 1.10

    # Group into rows
    rows, current_row, previous_y = [], [], None

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


def generate_valid_panels(rows, quads_x, quads_y):
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


def select_optimal_panels(panels, num_panels):
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


def process_panels(panels):
    """Process selected panels into final output format."""
    processed = []
    total_rad = 0.0

    for panel in panels:
        all_points = [pt for q in panel['quads'] for pt in q['original_coordinates']]
        min_x, max_x = np.min([p[0] for p in all_points]), np.max([p[0] for p in all_points])
        min_y, max_y = np.min([p[1] for p in all_points]), np.max([p[1] for p in all_points])
        avg_z = np.mean([p[2] for p in all_points])

        processed.append({
            'original_coordinates': [q['original_coordinates'] for q in panel['quads']],
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


def run_optimization(quads, panel_length, panel_width, num_panels, output_path):
    panel_placement = optimize_panel_placement(
        quads,
        num_panels,
        quad_size=1,
        panel_length=panel_length,
        panel_width=panel_width
    )
    
    visualize.visualize_quads_and_panels(quads, panel_placement)
    print('total radiance', panel_placement['total_radiance'])
    print('num_panels', len(panel_placement['panels']))


def read_results(path):
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
    comprehensive_results = read_results(sys.argv[4])

    run_optimization(
        comprehensive_results, 
        panel_length= int(sys.argv[2]),
        panel_width= int(sys.argv[3]),
        num_panels= int(sys.argv[1]),
        output_path=image_save
    )

    return True

if __name__ == '__main__':
    main()
