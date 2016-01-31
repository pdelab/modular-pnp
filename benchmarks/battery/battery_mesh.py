#! /usr/bin/python2.7

#from __future__ import print_function
from dolfin import *
from mshr import *
import numpy as np

class domain_spheres(SubDomain):
    # def __init__(self,_xc,_yc,_zc,_rc):
    #     self.xc = _xc
    #     self.yc = _yc
    #     self.zc = _zc
    #     self.rc = _rc
    def inside(self, x, on_boundary):
        flag=False
        for i in range(Numb_spheres):
            if on_boundary and np.abs((x[0]-xc[i])**2 + (x[1]-yc[i])**2 + (x[2]-zc[i])**2) < rc[i]**2+2.0:
                flag=True
        return flag

class domain_box(SubDomain):
    def inside(self, x, on_boundary):
        flag=False
        if on_boundary and np.abs(x[0]-Lx/2.0) < 1.0 :
                flag=True
        if on_boundary and np.abs(x[1]-Ly/2.0) < 1.0 :
                flag=True
        if on_boundary and np.abs(x[2]-Lz/2.0) < 1.0 :
                flag=True
        if on_boundary and np.abs(x[0]+Lx/2.0) < 1.0 :
                flag=True
        if on_boundary and np.abs(x[1]+Ly/2.0) < 1.0 :
                flag=True
        if on_boundary and np.abs(x[2]+Lz/2.0) < 1.0 :
                flag=True
        return flag

xc=[]
yc=[]
zc=[]
rc=[]

print "Reading the data..."
with open("./partlist.dat") as f:
    data = f.read()
data = data.split('\n')
n = len(data)

for row in data:
    line=row.split(" ")
    i=0
    for l in range(len(line)):
         if line[l]!='':
             if i==0:
                 xc.append(float(line[l])-25.0)
                 i+=1
             elif i==1:
                 yc.append(float(line[l])-25.0)
                 i+=1
             elif i==2:
                 zc.append(float(line[l])-25.0)
                 i+=1
             elif i==3:
                 rc.append(float(line[l]))
                 i+=1

Num_cells=50
Lx=100.0
Ly=100.0
Lz=100.0
domain = Box(Point(-Lx/2.0,-Ly/2.0,-Lz/2.0),Point(Lx/2.0,Ly/2.0,Lz/2.0))
print "Generating the domain..."
## The range can be at most len(x)=86
Numb_spheres=10
for i in range(10):
    domain = domain - Sphere(Point(xc[i],yc[i],zc[i]), rc[i])

print "Generating the mesh..."
mesh = generate_mesh(domain,Num_cells,"cgal")

# file = File("mesh.pvd")
# file << mesh
file = File("mesh.xml.gz")
file << mesh

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)
spheres = domain_spheres()
box = domain_box()
spheres.mark(boundary_parts, 1)
box.mark(boundary_parts, 2)

# file = File("boundary_parts.pvd")
# file << boundary_parts
file = File("boundary_parts.xml.gz")
file << boundary_parts
