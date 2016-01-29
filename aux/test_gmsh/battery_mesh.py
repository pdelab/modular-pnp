#! /usr/bin/python2.7

#from __future__ import print_function
from dolfin import *
from mshr import *

x=[]
y=[]
z=[]
r=[]

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
                 x.append(float(line[l])-25.0)
                 i+=1
             elif i==1:
                 y.append(float(line[l])-25.0)
                 i+=1
             elif i==2:
                 z.append(float(line[l])-25.0)
                 i+=1
             elif i==3:
                 r.append(float(line[l]))
                 i+=1

Num_cells=100
Lx=100.0
Ly=100.0
Lz=100.0
domain = Box(Point(-Lx/2.0,-Ly/2.0,-Lz/2.0),Point(Lx/2.0,Ly/2.0,Lz/2.0))
print "Generating the domain..."
## The range can be at most len(x)=86
for i in range(20):
    domain = domain - Sphere(Point(x[i],y[i],z[i]), r[i])

print "Generating the mesh..."
mesh = generate_mesh(domain,Num_cells,"cgal")

file = File("mesh.pvd")
file << mesh

# boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
