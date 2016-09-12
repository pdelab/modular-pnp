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

ref_Lx=7.2
ref_Ly=7.2
ref_Lz=7.2
for row in data:
    line=row.split(" ")
    i=0
    for l in range(len(line)):
         if line[l]!='':
             if i==0:
                 xc.append((float(line[l])-25.0)/7.2)
                 i+=1
             elif i==1:
                 yc.append((float(line[l])-25.0)/7.2)
                 i+=1
             elif i==2:
                 zc.append((float(line[l])-25.0)/7.2)
                 i+=1
             elif i==3:
                 rc.append((float(line[l]))/7.2)
                 i+=1

Num_cells=25
Lx=72.0/ref_Lx
Ly=72.0/ref_Ly
Lz=72.0/ref_Lz
domain = Box(Point(-Lx/2.0,-Ly/2.0,-Lz/2.0),Point(Lx/2.0,Ly/2.0,Lz/2.0))
print "Generating the domain..."
## The range can be at most len(x)=86
Numb_spheres=20
for i in range(Numb_spheres):
    domain = domain - Sphere(Point(xc[i],yc[i],zc[i]), rc[i])

print "Generating the mesh..."
mesh = generate_mesh(domain,Num_cells,"cgal")
mesh2 = generate_mesh(domain,Num_cells,"cgal")

p1=Point(Lx,0.0,0.0)
mesh2.translate(p1)
mm = MultiMesh(mesh, mesh2, 1)


file = File("mesh.xml.gz")
file << mesh
file2 = File("mesh.pvd")
file2 << mesh
file3 = File("mesh2.pvd")
file3 << mesh2

# Build multimesh
multimesh = MultiMesh()
multimesh.add(mesh)
multimesh.add(mesh2)
multimesh.build()

filemm = File("mm.pvd")
filemm << multimesh



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
