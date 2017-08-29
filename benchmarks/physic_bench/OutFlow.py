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
u2 = Function(V,"DATA_50/velocity_solution.xml")
u3 = Function(V,"DATA_100/velocity_solution.xml")
u4 = Function(V,"DATA_150/velocity_solution.xml")
u5 = Function(V,"DATA_200/velocity_solution.xml")
u6 = Function(V,"DATA_250/velocity_solution.xml")

E1 = 3.0/4.0*inner(u1,n)*ds(1)
E2 = 3.0/4.0*inner(u2,n)*ds(1)
E3 = 3.0/4.0*inner(u3,n)*ds(1)
E4 = 3.0/4.0*inner(u4,n)*ds(1)
E5 = 3.0/4.0*inner(u5,n)*ds(1)
E6 = 3.0/4.0*inner(u6,n)*ds(1)

v1 = assemble(E1)*1E3
v2 = assemble(E2)*1E3
v3 = assemble(E3)*1E3
v4 = assemble(E4)*1E3
v5 = assemble(E5)*1E3
v6 = assemble(E6)*1E3

y = np.array([v1,v2,v3,v4,v5,v6])
x = np.array([0,50,100,150,200,250])

print y[1:]
print x[1:]
print y[1:]/x[1:]

plt.figure(1)
plt.clf()
plt.plot(x, y,"-*")
# plt.plot(x, x*0.065/10.0,"--")
# plt.show()
plt.ylabel("Average outgoing flow in mm/s")
plt.xlabel("Voltage drop $\delta V$ in mV")
plt.savefig("OutFlow.eps")
plt.savefig("OutFlow.png")



N0=np.log(np.array([0.0192247,2.35473e-07,1.19454e-13])) # V=0
N2=np.log(np.array([0.0192246,2.31708e-07,1.92468e-13])) # V=50
N4=np.log(np.array([0.0191769,2.40639e-07,5.07067e-13,])) # V=100
N6=np.log(np.array([0.0191326,2.41349e-07,6.93888e-13])) # V=150
N8=np.log(np.array([0.0190566,2.48112e-07,5.19014e-13])) # V=200
N9=np.log(np.array([0.0189578,2.56926e-07,8.25984e-13]))  # V=250
xn =np.array([1,2,3])


plt.figure(1)
plt.clf()
plt.plot(xn, N0,"-*")
plt.plot(xn, N2,"-*")
plt.plot(xn, N4,"-*")
plt.plot(xn, N6,"-*")
plt.plot(xn, N8,"-*")
plt.plot(xn, N9,"-*")
# plt.plot(x, x*0.065/10.0,"--")
# plt.show()
plt.legend(("$\delta V=0$ mV","$\delta V=50$ mV","$\delta V=100$ mV","$\delta V=150$ mV","$\delta V=200$ mV","$\delta V=250$ mV"))
plt.xlabel("Newton Iteration")
plt.ylabel("log of the Relative Residual")
plt.savefig("Residual.eps")
plt.savefig("Residual.png")
