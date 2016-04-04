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

Num_cells=20
Lx=72.0/ref_Lx
Ly=72.0/ref_Ly
Lz=72.0/ref_Lz
domain = Box(Point(-Lx/2.0,-Ly/2.0,-Lz/2.0),Point(Lx/2.0,Ly/2.0,Lz/2.0))
print "Generating the domain..."
## The range can be at most len(x)=86
Numb_spheres=20

# Read mesh
mesh = Mesh("mesh.xml.gz")

# Function to mark inner surface of pulley
class SpheresSubDomain(SubDomain):
    def inside(self, x, on_boundary):
        flag=False
        for i in range(Numb_spheres):
            if (on_boundary and ( (x[0]-xc[i])**2 + (x[1]-yc[i])**2 + (x[2]-zc[i]**2) < (rc[i]**2)+2.0) ):
                flag=True
        return flag

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)
spheres = SpheresSubDomain()
spheres.mark(boundary_parts,1)

ds = Measure("ds", domain=mesh, subdomain_data=boundary_parts)
g=1.0
Cat=np.log(2.0)
An=1.0
qn=-1.0
qp=1.0
CG = FunctionSpace(mesh, "Lagrange", 1)
v = Function(CG)
v.interpolate(Constant(1.0))
# M = ( -(qp*exp(Cat) + qn*exp(An))*v )*dx + g*v*ds(1)
# print assemble(( -(qp*exp(Cat) + qn*exp(An))*v )*dx(mesh) + g*v*ds(1))
inte=assemble(( -(qp*exp(Cat) )*v )*dx(mesh) + g*v*ds(1))
print inte
An=np.log(inte/assemble(qn*v*dx(mesh)))
print Cat, An
print assemble(( -(qp*exp(Cat) + qn*exp(An))*v )*dx(mesh) + g*v*ds(1))
# print assemble(g*v*ds(1))
