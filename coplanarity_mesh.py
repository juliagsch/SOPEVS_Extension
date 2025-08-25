import numpy as np
import open3d as o3d
import trimesh

from dataclasses import dataclass
from raytracing import ray_polygon_intersection, triangulate_polygon
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
    """

    def __init__(self,
                 building_V: np.ndarray,
                 building_F : List[List[int]],
                 surroundings_V: np.ndarray,
                 surroundings_F: List[List[int]],
                 grid_size: float = 1.0,
                 target_faces: int = 100):

        self.building_V = building_V
        self.building_F  = building_F 
        self.surroundings_V = surroundings_V
        self.surroundings_F = surroundings_F
        self.grid_size = grid_size
        self.target_faces = target_faces

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

    def identify_rooftops(self, max_tilt=60):
        """
        Identify rooftop faces based on:
        1. Faces with a tilt smaller than max_tilt degrees
        2. Whether they are the first face to be intersected by vector (0,0,-1) which determines if they would be hit by rain.
        """
        ray_dir = np.array([0, 0, 1], dtype=float)

        # Filter vertical faces
        roof_faces = []
        for face in self.building_F :
            # Calculate face normal
            face_v = self.building_V[face]
            normal = self.compute_normal(face_v)
            tilt = np.degrees(np.arccos(normal[2]))

            # Consider only faces with a tilt smaller than max_tilt degrees
            if tilt > max_tilt: 
                continue
            
            sample_points = np.mean(face_v, axis=0) + np.array(face_v)
            sample_point_occluded = [False for _ in range(len(sample_points))]

            for other_face in self.building_F :
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
        Check if points reside inside the polygon
        """
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            if self.is_point_on_segment(px, py, x1, y1, x2, y2, tol):
                return True

        inside = False
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            if (y1 > py) != (y2 > py):
                dy = y2 - y1
                if abs(dy) < tol:
                    continue
                t = (py - y1) / dy
                x_intersect = x1 + t * (x2 - x1)
                if px <= x_intersect + tol:
                    inside = not inside

        return inside


    def is_point_on_segment(self, px, py, x1, y1, x2, y2, tol=1e-8):
        cross_product = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)
        if abs(cross_product) > tol:
            return False
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
        for u0, v0, u1, v1 in squares_uv:
            corners_uv = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
            corners_3d = [ (A + u*u_vec + v*v_vec).tolist() for u, v in corners_uv ]
            mesh_squares.append(corners_3d)

        return mesh_squares


    def process_roof_segment(self, face, idx):
        tilt, azimuth = self.calculate_tilt_azimuth(self.building_V[face])
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

        mesh = trimesh.Trimesh(vertices=np.array(self.surroundings_V), faces=(np.array(filtered_faces)))
        o3d_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(mesh.vertices),
            triangles=o3d.utility.Vector3iVector(mesh.faces)
        )
        # Simplify
        simplified_o3d = o3d_mesh.simplify_quadric_decimation(self.target_faces)

        self.surroundings_V = np.asarray(simplified_o3d.vertices)
        self.surroundings_F = np.asarray(simplified_o3d.triangles).tolist()

        num_v = len(self.surroundings_V)
        for building_poly in self.building_F:
            triangles = triangulate_polygon(building_poly)
            for tri in triangles:
                self.surroundings_F.append(np.array(tri)+num_v)

        self.surroundings_V = np.vstack([self.surroundings_V, self.building_V])
        # mesh = trimesh.Trimesh(vertices=np.array(self.surroundings_V), faces=(np.array(self.surroundings_F)))
        # mesh.show()

