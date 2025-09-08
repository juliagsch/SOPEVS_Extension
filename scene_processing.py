import networkx as nx
import numpy as np
import open3d as o3d
import trimesh

from dataclasses import dataclass
from raytracing import ray_polygon_intersection, ray_triangle_intersection
from typing import List

@dataclass
class RoofSegment:
    idx: int
    tilt: float
    azimuth: float
    panels: List[List[float]]
    polygon: List[List[float]]

class Scene:
    """
    Attributes:
        building_V (np.ndarray): Nx3 array of vertices (x,y,z coordinates)
        building_F  (List[List[int]]): List of face vertex indices

        surroundings_V (np.ndarray): Nx3 array of vertices (x,y,z coordinates)
        surroundings_F (List[List[int]]): List of face vertex indices

        grid_size (float): Size in m^2 for each grid cell
        target_faces (int): Number of occlusion faces to include
        angle_tolerance (float): Angle tolerance in degrees for merging faces
        max_roof_tilt (float): Maximum roof tilt in degrees to consider a face as rooftop
    """

    def __init__(self,
                 building_V: np.ndarray,
                 building_F : List[List[int]],
                 surroundings_V: np.ndarray,
                 surroundings_F: List[List[int]],
                 grid_size: float = 1.0,
                 target_faces: int = 100,
                 angle_tolerance: float = 1.0,
                 max_roof_tilt: float = 60.0):

        self.building_V = building_V
        self.building_F  = building_F 
        self.surroundings_V = surroundings_V
        self.surroundings_F = surroundings_F
        self.grid_size = grid_size
        self.target_faces = target_faces
        self.angle_tolerance = angle_tolerance
        self.max_roof_tilt = max_roof_tilt

        self.united_faces = self.merge_faces(self.building_V, self.building_F)

        # Identify and filter roof faces
        self.roof_faces = self.identify_rooftops()

        # Process roof segments
        self.roof_segments = [self.process_roof_segment(face, idx) for idx, face in enumerate(self.roof_faces)]

        # Get lowest point on roof
        self.min_z_roof = np.min([z for segment in self.roof_segments for z in segment.polygon[:,2]])

        # Shrink surroundings mesh and add building faces for self-shading analysis
        self.simplify_surroundings()
        
        # Get surroundings triangles for easier processing later on
        self.surroundings_triangles = [[self.surroundings_V[face[0]], self.surroundings_V[face[1]], self.surroundings_V[face[2]]] for face in self.surroundings_F]

        # Get all panels for plots
        self.panels = []
        for roof_segment in self.roof_segments:
            for panel in roof_segment.panels:
                self.panels.append(panel)



    def shared_edge(self, poly, tri):
        common = [v for v in poly if any(np.allclose(v, u) for u in tri)]
        if len(common) == 2:
            return common
        return None


    def merge_faces(self, vertices, faces, min_area=1.0):
        """Unite faces with similar normals and vertices to convert triangular mesh to polygonal mesh."""
        mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=(np.array(faces)))
        normals = mesh.face_normals

        # Group faces based on normal similarity
        angle_tolerance = np.radians(self.angle_tolerance)
        adjacency = mesh.face_adjacency
        G = nx.Graph()
        G.add_nodes_from(range(len(mesh.faces)))

        for f1, f2 in adjacency:
            angle = np.arccos(np.clip(np.dot(normals[f1], normals[f2]), -1.0, 1.0))
            if angle < angle_tolerance:
                G.add_edge(f1, f2)

        components = list(nx.connected_components(G))

        face_areas = mesh.area_faces

        # Filter out components with 1 face and area < min_area square m
        filtered_components = []
        for comp in components:
            if len(comp) > 1:
                filtered_components.append(comp)
            else:
                face_idx = list(comp)[0]
                if face_areas[face_idx] >= min_area:
                    filtered_components.append(comp)

        merged_faces = []

        for comp in filtered_components:
            face_indices = list(comp)
            merged = list(mesh.faces[face_indices[0]])
            remaining_faces = face_indices[1:]

            while remaining_faces:

                merged_this_round = False
                for face_idx in remaining_faces:
                    face = mesh.faces[face_idx]
                    common_edge = self.shared_edge(merged, list(face))  # should return two shared vertex indices

                    if common_edge is not None:
                        # Find the vertex in the triangle that's NOT part of the common edge
                        not_common = [v for v in face if v not in common_edge]

                        if len(not_common) != 1:
                            raise Exception("Exactly one vertex should not be common when merging a triangle with a polygon")

                        # Insert the non-shared vertex into the merged polygon in the correct place
                        idx0 = merged.index(common_edge[0])
                        idx1 = merged.index(common_edge[1])

                        # Decide insertion point: after idx0 if idx1 == idx0+1 or after idx1 if idx0==idx1+1
                        if (idx1 - idx0) % len(merged) == 1:
                            insert_idx = idx1
                        else:
                            insert_idx = idx0
                        
                        # Handle case where vertex should be appended at the end of the list.
                        if idx1 == 0 and idx0!=1:
                            insert_idx = idx0+1
                        if idx0 == 0 and idx1!=1:
                            insert_idx = idx1+1

                        merged.insert(insert_idx, not_common[0])
                        remaining_faces.remove(face_idx)
                        merged_this_round = True
                        break

                if not merged_this_round:
                    break
            
            merged_faces.append(merged)
        return merged_faces


    def identify_rooftops(self):
        """
        Identify rooftop faces based on:
        1. Faces with a tilt smaller than max_tilt degrees
        2. Whether they are the first face to be intersected by vector (0,0,-1) which determines if they would be hit by rain.
        """
        ray_dir = np.array([0, 0, 1], dtype=float)

        # Filter vertical faces
        roof_faces = []
        for face in self.united_faces:
            # Calculate face normal
            face_v = self.building_V[face]
            normal = self.compute_normal(face_v)
            tilt = np.degrees(np.arccos(normal[2]))

            # Consider only faces with a tilt smaller than max_tilt degrees
            if tilt > self.max_roof_tilt: 
                continue
            
            # Check if reachable by rain from above.
            sample_points = [np.mean(face_v, axis=0)]
            sample_points.extend(face_v)
            sample_point_occluded = [False for _ in range(len(sample_points))]

            for other_face in self.united_faces:
                polygon = self.building_V[other_face]
                for idx, sample in enumerate(sample_points):
                    intersects = ray_polygon_intersection(sample + 0.1*ray_dir, ray_dir, polygon)
                    if intersects:
                        sample_point_occluded[idx] = True

            # If there are some points on the face which could be reached by rain we classify the face as a roof segement
            if False in sample_point_occluded:
                roof_faces.append(face)
        
        return roof_faces


    def compute_normal(self, polygon):
        """Compute normal for shapes with more than 3 vertices."""
        points = np.array(polygon)
        centroid = np.mean(points, axis=0)
        shifted = points - centroid
        _, _, Vt = np.linalg.svd(shifted)
        normal = Vt[2]
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            return None
        # Make sure that the normal points upwards
        if normal[2] < 0:
            normal = -normal
        return normal / norm

    def point_in_polygon(self, px, py, polygon, tol=1e-8):
        """
        Check if a point (px, py) resides inside a polygon.

        Args:
            px (float): x-coordinate of the point.
            py (float): y-coordinate of the point.
            polygon (list of tuples): List of (x, y) coordinates representing the polygon vertices.
            tol (float): Tolerance for numerical comparisons.

        Returns:
            bool: True if the point is inside the polygon or on its boundary, False otherwise.
        """
        n = len(polygon)
        # Check if the point lies on any of the polygon's edges
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            if self.is_point_on_segment(px, py, x1, y1, x2, y2, tol):
                return True

        # Use the ray-casting algorithm to determine if the point is inside the polygon
        inside = False
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            # Check if the ray intersects the edge
            if (y1 > py) != (y2 > py):
                dy = y2 - y1
                if abs(dy) < tol:
                    continue
                t = (py - y1) / dy
                x_intersect = x1 + t * (x2 - x1)
                # Toggle the inside flag if the ray intersects the edge
                if px <= x_intersect + tol:
                    inside = not inside

        return inside

    def is_point_on_segment(self, px, py, x1, y1, x2, y2, tol=1e-8):
        """
        Check if a point (px, py) lies on a line segment defined by (x1, y1) and (x2, y2).

        Args:
            px (float): x-coordinate of the point.
            py (float): y-coordinate of the point.
            x1 (float): x-coordinate of the first endpoint of the segment.
            y1 (float): y-coordinate of the first endpoint of the segment.
            x2 (float): x-coordinate of the second endpoint of the segment.
            y2 (float): y-coordinate of the second endpoint of the segment.
            tol (float): Tolerance for numerical comparisons.

        Returns:
            bool: True if the point lies on the segment, False otherwise.
        """
        # Check if the point is collinear with the segment
        cross_product = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)
        if abs(cross_product) > tol:
            return False

        # Check if the point lies within the segment bounds
        min_x = min(x1, x2) - tol
        max_x = max(x1, x2) + tol
        min_y = min(y1, y2) - tol
        max_y = max(y1, y2) + tol
        return (px >= min_x and px <= max_x) and (py >= min_y and py <= max_y)


    def calculate_tilt_azimuth(self, polygon):
        """Calculate surface orientation from triangle geometry with type safety"""
        # Convert to numpy array if needed
        if not isinstance(polygon, np.ndarray):
            polygon = np.array(polygon, dtype=np.float64)

        # Calculate normal vector
        normal = self.compute_normal(polygon)

        # Normalize and calculate angles
        tilt = np.degrees(np.arccos(normal[2]))
        azimuth = np.degrees(np.arctan2(normal[1], normal[0])) # Counter clock-wise, 0 degrees pointing East
        azimuth = (360-azimuth) % 360 # Convert to clock-wise
        azimuth = (azimuth + 90) % 360 # Convert from 0 degrees pointing East to 0 degrees pointing North

        if np.isclose(tilt, 0.0): # Prevent errors if flat roof
            azimuth = 180.0 
        return tilt, azimuth


    def generate_check_points(self, u, v, u_end, v_end, samples=10):
        """Generate interior test points on a grid."""
        xs = np.linspace(u, u_end, samples)
        ys = np.linspace(v, v_end, samples)
        return [(x, y) for x in xs for y in ys]

    def validate_panel(self, panel):
        """Check if a panel intersects with any building face."""
        # Check edges for face intersections
        for i in range(len(panel)):
            v_start = np.array(panel[i])
            v_end = np.array(panel[(i + 1) % len(panel)])
            edge_dir = v_end - v_start
            edge_length = np.linalg.norm(edge_dir)
            edge_dir /= edge_length

            for tri in self.building_F:
                intersects, t = ray_triangle_intersection(v_start, edge_dir, self.building_V[tri])
                if intersects and 0 < t < edge_length:
                    return False
        # Check diagonal for face intersections
        for i in range(2):
            v_start = np.array(panel[i])
            v_end = np.array(panel[(i + 2) % len(panel)])
            edge_dir = v_end - v_start
            edge_length = np.linalg.norm(edge_dir)
            edge_dir /= edge_length 

            for tri in self.building_F:
                intersects, t = ray_triangle_intersection(v_start, edge_dir, self.building_V[tri])
                if intersects and 0 < t < edge_length:
                    return False
        
        # Check if reachable by sun from above. 
        # We handle the edge case when panels are placed within a building chimney or similar.
        # Additionally, panels occluded from above have a very low energy yield and are not relevant for the optimization.
        sample_points = [np.mean(panel, axis=0)]
        sample_points.extend(panel)
        sample_point_occluded = [False for _ in range(len(sample_points))]

        for face in self.building_F:
            for idx, sample in enumerate(sample_points):
                intersects,_ = ray_triangle_intersection(sample, [0,0,1], self.building_V[face])
                if intersects:
                    sample_point_occluded[idx] = True
        if not (False in sample_point_occluded):
            return False
        return True

    def generate_grid(self, face):
        """
        Divide a roof face into square tiles of size grid_size.
        Returns a list of squares, each as a list of 4 corner points.
        """
        # Get vertices and face normal
        face_verts = self.building_V[face]
        normal = self.compute_normal(face_verts)
        if normal is None:
            return []

        # Choose a reference edge (lowest 2 vertices in z)
        face_z = sorted(face_verts, key=lambda v: v[2])
        lowest_two = face_z[:2]
        A, B = np.array(lowest_two[0]), np.array(lowest_two[1])

        u_vec = B - A
        if np.linalg.norm(u_vec) < 1e-6:
            return []
        u_vec /= np.linalg.norm(u_vec)

        v_vec = np.cross(normal, u_vec)
        if np.linalg.norm(v_vec) < 1e-6:
            return []
        v_vec /= np.linalg.norm(v_vec)

        # Project all face vertices into local UV space
        uv_coords = []
        for idx in face:
            vertex = np.array(self.building_V[idx])
            rel = vertex - A
            u = np.dot(rel, u_vec)
            v = np.dot(rel, v_vec)
            uv_coords.append((u, v))

        u_vals = [u for u, _ in uv_coords]
        v_vals = [v for _, v in uv_coords]
        u_min, u_max = min(u_vals), max(u_vals)
        v_min, v_max = min(v_vals), max(v_vals)

        # Tile UV bounding box with grid squares 
        squares_uv = []
        step = self.grid_size
        min_dim = 0.3 * step   # ignore tiny slivers at the boundary

        u = u_min
        while u < u_max:
            v = v_min
            while v < v_max:
                u_end = min(u + step, u_max)
                v_end = min(v + step, v_max)

                # Skip too small
                if (u_end - u) < min_dim or (v_end - v) < min_dim:
                    v += step
                    continue

                # Check if square is fully inside polygon
                check_points = self.generate_check_points(u, v, u_end, v_end, samples=3)
                if all(self.point_in_polygon(x, y, uv_coords) for x, y in check_points):
                    squares_uv.append((u, v, u_end, v_end))

                v += step
            u += step

        # Convert accepted UV squares back into 3D
        mesh_squares = []
        building_mesh = trimesh.Trimesh(vertices=np.array(self.building_V), faces=(np.array(self.building_F)))

        for u0, v0, u1, v1 in squares_uv:
            corners_uv = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
            corners_3d = [ (A + u*u_vec + v*v_vec).tolist() for u, v in corners_uv ]

            # Offset corners_3d by 0.1 on the z-axis to avoid self-intersection
            panel_polygon = [[x, y, z + 1] for x, y, z in corners_3d]
            # Check for intersection with any face in building_V
            if self.validate_panel(panel_polygon):
                mesh_squares.append(corners_3d)

        return mesh_squares


    def process_roof_segment(self, face, idx):
        tilt, azimuth = self.calculate_tilt_azimuth(self.building_V[face])
        # print(tilt, azimuth)
        mesh_squares = self.generate_grid(face)
    
        return RoofSegment(
            idx=idx,
            tilt=tilt,
            azimuth=azimuth,
            panels=mesh_squares,
            polygon=self.building_V[face]
        )
    
    def filter_faces(self, V, F, min_z):
        filtered_faces = []
        for face in F:
            if np.any(V[face][:,2]>min_z):
                filtered_faces.append(face)
        return filtered_faces
    

    def simplify_surroundings(self):
        filtered_faces = self.filter_faces(self.surroundings_V, self.surroundings_F, self.min_z_roof)
        # visualize.plot_surroundings(self.surroundings_V, filtered_faces)

        mesh = trimesh.Trimesh(vertices=np.array(self.surroundings_V), faces=(np.array(filtered_faces)))
        o3d_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(mesh.vertices),
            triangles=o3d.utility.Vector3iVector(mesh.faces)
        )
        # Simplify
        simplified_o3d = o3d_mesh.simplify_quadric_decimation(self.target_faces)

        self.surroundings_V = np.asarray(simplified_o3d.vertices)
        self.surroundings_F = np.asarray(simplified_o3d.triangles).tolist()
        # visualize.plot_surroundings(self.surroundings_V, self.surroundings_F)


        num_v = len(self.surroundings_V)
        for tri in self.building_F:
            self.surroundings_F.append(np.array(tri)+num_v)

        self.surroundings_V = np.vstack([self.surroundings_V, self.building_V])
        # visualize.plot_surroundings(self.surroundings_V, self.surroundings_F)

        # mesh = trimesh.Trimesh(vertices=np.array(self.surroundings_V), faces=(np.array(self.surroundings_F)))
        # mesh.show()

