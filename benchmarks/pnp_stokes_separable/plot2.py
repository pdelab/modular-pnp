#! /usr/bin/python2.7

#from __future__ import print_function
from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


# Fo = "./meshOut_2/"
Fo = "./output2/"

# Set backend to PETSC
# parameters['linear_algebra_backend'] = 'Eigen'
# parameters["linear_algebra_backend"] = "PETSc"
parameters["allow_extrapolation"] = True


# Load mesh and define function space
mesh = Mesh("mesh.xml.gz");
CG = FunctionSpace(mesh, "CG", 1)
RT = FunctionSpace(mesh, "RT", 1)
phi = Function(CG,Fo+"potential.xml")
u = Function(RT,Fo+"velocity.xml")

# Plot sigma and u
plot(u)
interactive()
