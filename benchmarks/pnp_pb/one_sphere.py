#! /usr/bin/python2.7

#from __future__ import print_function
from dolfin import *
from mshr import *
import numpy as np

rc=0.1

Num_cells=35
Lx=0.32
Ly=0.32
Lz=0.32
domain = Box(Point(-Lx/2.0,-Ly/2.0,-Lz/2.0),Point(Lx/2.0,Ly/2.0,Lz/2.0))
print "Generating the domain..."
## The range can be at most len(x)=86


domain = domain - Sphere(Point(0.0,0.0,0.0), rc)

print "Generating the mesh..."
mesh = generate_mesh(domain,Num_cells,"cgal")

file = File("mesh1.pvd")
file << mesh
file = File("mesh1.xml.gz")
file << mesh
