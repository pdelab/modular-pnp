#! /usr/bin/python2.7

"""Solving
    u_t - u_xx + 1/Epsilon**2*(u^2-1)^2 = 0
"""

#from __future__ import print_function
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy.random as rd
import os
import sys

parameters["linear_algebra_backend"] = "PETSc"
parameters["allow_extrapolation"] = True

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0)

def boundary(x,on_boundary):
    return (abs(x[0]-1.0)<1E-6) and on_boundary

mesh = Mesh("mesh3.xml.gz")
V = FunctionSpace(mesh, "RT", 1)

left = Left()
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
left.mark(boundaries, 1)
# ds = Measure("ds")[boundaries]
ds = ds(subdomain_data = boundaries)
n = FacetNormal(mesh)

u1 = Function(V,"DATA_0/velocity_solution.xml")
u2 = Function(V,"DATA_2/velocity_solution.xml")
u3 = Function(V,"DATA_4/velocity_solution.xml")
u4 = Function(V,"DATA_6/velocity_solution.xml")
u5 = Function(V,"DATA_8/velocity_solution.xml")


E1 = inner(u1,n)*ds(1)
E2 = inner(u2,n)*ds(1)
E3 = inner(u3,n)*ds(1)
E4 = inner(u4,n)*ds(1)
E5 = inner(u5,n)*ds(1)

v1 = assemble(E1)
v2 = assemble(E2)
v3 = assemble(E3)
v4 = assemble(E4)
v5 = assemble(E5)

y = np.array([v1,v2,v3,v4,v5])
x = np.array([0,2,4,6,8])

plt.figure(1)
plt.clf()
plt.plot(x, y,"-*")
# plt.plot(x, x*0.065/10.0,"--")
# plt.show()
plt.ylabel("Average outgoing flow")
plt.xlabel("Voltage drop $\delta V$")
plt.savefig("OutFlow.eps")
plt.savefig("OutFlow.png")



N0=np.log(np.array([0.0192247,2.35473e-07,1.19454e-13])) # V=0
N2=np.log(np.array([0.0192071,2.40205e-07,3.43298e-13])) # V=2
N4=np.log(np.array([0.0191623,2.39543e-07,3.38475e-13])) # V=4
N6=np.log(np.array([0.0191008,2.37633e-07,1.88464e-12])) # V=6
N8=np.log(np.array([0.018905,2.48055e-07,1.84684e-12])) # V=8
xn =np.array([1,2,3])


plt.figure(1)
plt.clf()
plt.plot(xn, N0,"-*")
plt.plot(xn, N2,"-*")
plt.plot(xn, N4,"-*")
plt.plot(xn, N6,"-*")
plt.plot(xn, N8,"-*")
# plt.plot(x, x*0.065/10.0,"--")
# plt.show()
plt.legend(("$\delta V=0$","$\delta V=2$","$\delta V=4$","$\delta V=6$","$\delta V=8$"))
plt.xlabel("Newton Iteration")
plt.ylabel("log of the Relative Residual")
plt.savefig("Residual.eps")
plt.savefig("Residual.png")
