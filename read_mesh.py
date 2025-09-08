import numpy as np
import trimesh


def read_building(filename):
    '''
    Used to read a building mesh from filename.obj 
    The building mesh should not be triangulated as it is assumed that one roof segment consists of one face.
    '''
    # Load the mesh using trimesh
    mesh = trimesh.load(filename, process=True)
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # Calculate the minimum coordinates for offset
    min_x = np.min(verts[:, 0])
    min_y = np.min(verts[:, 1])
    min_z = np.min(verts[:, 2])

    # Apply the offset to the vertices
    verts -= np.array([min_x, min_y, min_z])
    offset = [min_x, min_y, min_z]
    
    return verts, faces, offset


def read_surroundings(filename, offset=[0,0,0]):
    '''
    Used to read a surroundings mesh from filename.obj. The offset is applied for alignment with the building mesh.
    A triangular mesh is expected.
    '''
    mesh = trimesh.load(filename)
    verts = np.array(mesh.vertices)
    faces = mesh.faces

    for i in range(len(verts)):
        verts[i,:] = verts[i,0]-offset[0], verts[i,1]-offset[1], verts[i,2]-offset[2]
    return verts, faces
