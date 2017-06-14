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
                 xc.append((float(line[l])-25.0)/36.0)
                 i+=1
             elif i==1:
                 yc.append((float(line[l])-25.0)/36.0)
                 i+=1
             elif i==2:
                 zc.append((float(line[l])-25.0)/36.0)
                 i+=1
             elif i==3:
                 rc.append((float(line[l]))/36.0)
                 i+=1

print np.max(xc)
print np.max(rc)


Num_cells=15
Lx=2.2
Ly=2.2
Lz=2.2
domain = Box(Point(-Lx/2.0,-Ly/2.0,-Lz/2.0),Point(Lx/2.0,Ly/2.0,Lz/2.0))
print "Generating the domain..."
## The range can be at most len(x)=86
Numb_spheres=3
for i in range(Numb_spheres):
    domain = domain - Sphere(Point(xc[i],yc[i],zc[i]), rc[i])

print "Generating the mesh..."
mesh = generate_mesh(domain,Num_cells,"cgal")

file = File("mesh.pvd")
file << mesh
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
# file = File("boundary_parts.xml.gz")
# file << boundary_parts

Ns=len(rc)
f = open("values.txt",'w')

f.write("double xc["+str(len(xc))+"] = { ")
for i in range(Ns-1):
    f.write(str(xc[i])+",")
    if i%10==0 and i!=0:
        f.write("\n\t")
f.write(str(xc[Ns-1])+"};\n\n")

f.write("double yc["+str(len(xc))+"] = { ")
for i in range(Ns-1):
    f.write(str(yc[i])+",")
    if i%10==0 and i!=0:
        f.write("\n\t")
f.write(str(yc[Ns-1])+"};\n\n")

f.write("double zc["+str(len(xc))+"] = { ")
for i in range(Ns-1):
    f.write(str(zc[i])+",")
    if i%10==0 and i!=0:
        f.write("\n\t")
f.write(str(zc[Ns-1])+"};\n\n")

f.write("double rc["+str(len(xc))+"] = { ")
for i in range(Ns-1):
    f.write(str(rc[i])+",")
    if i%10==0 and i!=0:
        f.write("\n\t")
f.write(str(rc[Ns-1])+"};\n\n")
