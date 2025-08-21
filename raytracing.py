import numpy as np

def triangulate_polygon(polygon):
    """Applies Fan Triangulation"""
    n = len(polygon)
    if n < 3:
        return []
    if n == 3:
        return polygon

    triangles = []
    for i in range(1, n-1):
        triangles.append([polygon[0], polygon[i], polygon[i + 1]])
    return triangles


def ray_polygon_intersection(ray_origin, ray_dir, polygon):
    """Triangulates polygon and then calls the Möller-Trumbore Algorithm"""
    triangles = triangulate_polygon(polygon)
    for triangle in triangles:
        if ray_triangle_intersection(ray_origin, ray_dir, triangle):
            return True
    
    return False


def ray_triangle_intersection(ray_origin, ray_dir, triangle, epsilon=1e-10):
    """Möller-Trumbore algorithm with dynamic epsilon"""
    v0, v1, v2 = [np.array(p) for p in triangle]
    edge1 = v1 - v0
    edge2 = v2 - v0

    h = np.cross(ray_dir, edge2)
    a = np.dot(edge1, h)

    if abs(a) < epsilon:
        return False

    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return False

    q = np.cross(s, edge1)
    v = f * np.dot(ray_dir, q)

    if v < 0.0 or u + v > 1.0:
        return False

    t = f * np.dot(edge2, q)
    return t >= epsilon