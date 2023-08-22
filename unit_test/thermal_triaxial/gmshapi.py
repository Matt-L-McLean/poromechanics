from mpi4py import MPI
import gmsh
import numpy as np

def mesh_cylindrical_sample(name="Triaxial",d=25e-3,gdim=3,o=1,lc=5e-3,comm=MPI.COMM_WORLD):
    # d --> sample diameter [m]
    # 0 --> mesh order
    # Mesh a cylindrical rock sample that is typically used in triaxial tests
    if comm.rank == 0:
        gmsh.initialize()
        gmsh.model.add(name)
    
        h = 2.5*d # sample height [m]
        
        cyl = gmsh.model.occ.addCylinder(0,0,0,0,0,h,d/2)
        
        gmsh.model.occ.synchronize()
        
        points, lines, surfaces, volumes = [gmsh.model.getEntities(d) for d in [0, 1, 2, 3]]
        
        gmsh.model.mesh.setSize(points, lc)
        
        gmsh.model.mesh.generate()

        gmsh.model.mesh.setOrder(o)
        
        gmsh.model.addPhysicalGroup(gdim, [1], 1)
        gmsh.model.setPhysicalName(gdim, 1, 'Omega')
        
        gmsh.model.addPhysicalGroup(gdim-1, [1], 2)
        gmsh.model.setPhysicalName(gdim-1, 2, 'Gamma_s')
        
        gmsh.model.addPhysicalGroup(gdim-1, [2], 3)
        gmsh.model.setPhysicalName(gdim-1, 3, 'Gamma_t')
        
        gmsh.model.addPhysicalGroup(gdim-1, [3], 4)
        gmsh.model.setPhysicalName(gdim-1, 4, 'Gamma_b')
        
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(gdim, 1)
        numElem = sum(len(i) for i in elemTags)
        
    return gmsh.model if comm.rank == 0 else None, gdim, h, numElem