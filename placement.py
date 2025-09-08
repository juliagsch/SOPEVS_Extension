import numpy as np
import os
import json
from dataclasses import dataclass


@dataclass
class CandidatePanel:
    total_production: float
    quads: list


def panel_placement(quads, num_panels, quad_size, panel_height, panel_width, min_segment_size=6):
    """Solar panel placement based on radiance values."""
    segments = {}
    # Filter out small segments
    for key, quad in zip(quads.keys(), quads.values()):
        segment_id = key.split('_')[0]
        if segment_id not in segments:
            segments[segment_id] = 0
        segments[segment_id] += 1
    
    for segment_id, count in segments.items():
        if count < min_segment_size:
            for key in list(quads.keys()):
                if key.split('_')[0] == segment_id:
                    del quads[key]

    # Calculate panel dimensions in quad units
    quads_x = max(1, int(np.ceil(panel_width / quad_size)))
    quads_y = max(1, int(np.ceil(panel_height / quad_size)))

    # Prepare grid structure
    candidates = get_panel_combinations(quads, quads_x, quads_y)
    # If not square, also check the other orientation
    if quads_x != quads_y:
        candidates += get_panel_combinations(quads, quads_y, quads_x)
    return select_panels(candidates, num_panels)


def get_panel_combinations(quads, quads_x, quads_y):
    """Generate all valid panel placements."""
    candidates = []
    
    for key, quad in zip(quads.keys(), quads.values()):
        valid_panel = True
        candidate = CandidatePanel(total_production=quad['average_radiance'], quads=[key])
        column = [key]
        # Extend in y direction
        for _ in range(quads_y-1):
            found_next = False
            current = quads[column[-1]]
            edge = [current['original_coordinates'][2],current['original_coordinates'][3]]
            for key_next, quad_next in zip(quads.keys(), quads.values()):
                if key_next == column[-1]:
                    continue
                if np.allclose(quad_next['original_coordinates'][1], edge[0]) and np.allclose(quad_next['original_coordinates'][0], edge[1]):
                    candidate.total_production += quad_next['average_radiance']
                    candidate.quads.append(key_next)
                    column.append(key_next)
                    found_next = True
                    break
            if not found_next:
                valid_panel = False
                break

        # Extend in x direction
        for col_start in column:
            current_row = [col_start]
            for _ in range(quads_x-1):
                found_next = False
                current = quads[current_row[-1]]
                edge = [current['original_coordinates'][0], current['original_coordinates'][3]]
                for key_next, quad_next in zip(quads.keys(), quads.values()):
                    if key_next == current_row[-1]:
                        continue
                    if np.allclose(quad_next['original_coordinates'][1], edge[0]) and np.allclose(quad_next['original_coordinates'][2], edge[1]):
                        candidate.total_production += quad_next['average_radiance']
                        candidate.quads.append(key_next)
                        current_row.append(key_next)
                        found_next = True
                        break
                if not found_next:
                    valid_panel = False
                    break
            if not valid_panel:
                break

        if valid_panel:
            candidate.total_production = candidate.total_production / (quads_x * quads_y) * (8760 / 1000)
            candidates.append(candidate)

    return candidates


def select_panels(panel_candidates, num_panels):
    """Select non-overlapping panels using greedy algorithm."""
    # Sort candidates by total production
    panels = sorted(panel_candidates, key=lambda x: x.total_production, reverse=True)
    selected = []
    used_quads = set()

    for panel in panels:
        if len(selected) >= num_panels:
            break

        conflict = any(q in used_quads for q in panel.quads)
        if not conflict:
            selected.append(panel)
            used_quads.update(q for q in panel.quads)
    return selected


def read_results(path):
    filename = os.path.join(path, "comprehensive_results.txt")
    try:
        with open(filename, 'r') as f:
            content = f.read().strip()

        data = json.loads(content)
        return data
    except Exception as e:
        print(f"Error reading results: {e}")
        return None

