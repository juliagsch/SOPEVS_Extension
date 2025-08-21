import numpy as np
import trimesh


def read_building(filename):
    '''
    Used to read a building mesh from filename.obj 
    The building mesh should not be triangulated as it is assumed that one roof segment consists of one face.
    '''
    if '.obj' not in filename:
        raise Exception('Please pass an .obj file.')
    
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    faces, verts = [], []

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
    min_x = min(verts[:,0])
    min_y = min(verts[:,1])
    min_z = min(verts[:,2])

    for i in range(len(verts)):
        verts[i,:] = verts[i,0]-min_x, verts[i,1]-min_y, verts[i,2]-min_z
    offset = [min_x, min_y, min_z]
    
    return verts, faces, offset


def read_surroundings(filename, offset=[0,0,0]):
    '''
    Used to read a surroundings mesh from filename.obj. The offset is applied for alignment with the building mesh.
    Triangulation is expected.
    '''
    mesh = trimesh.load(filename)
    verts = np.array(mesh.vertices)
    faces = mesh.faces

    for i in range(len(verts)):
        verts[i,:] = verts[i,0]-offset[0], verts[i,1]-offset[1], verts[i,2]-offset[2]
    return verts, faces
