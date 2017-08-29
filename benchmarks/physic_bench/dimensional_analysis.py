#! /usr/bin/python2.7

from dolfin import *
import numpy as np

ec = 1.6021766209*10.0**(-19) #C (elementary charge)
kB = 1.38064853*10**(-23) # J*K-1 (Boltzmann constant)
Epsilon_0  = 8.854187817 * 10**(-12) #  C V-1 m-1 (vacuum permittivity)
T = 300.0 # K
Epsilon = 80*Epsilon_0 # in water


Phi_ref = ec /( kB * T)
rho_ref = 6.02214085774*10**(23)*100
L=100
Epsilon_ref = 80*Epsilon_0*kB*T/(ec**2*rho_ref*L**2)

## This should give you Epsilon Ref
print -12-23+2*19-25
print 80*8.854187817*1.38064853*300/(1.6021766209**2*6.02214085774*10**2)

print Epsilon_ref

Fo = "./output_PNP/"

mesh = Mesh(Fo+"accepted_mesh.xml.gz")
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
# TH = P1 * P1 * P1
TH = MixedElement([P1, P1, P1])
V = FunctionSpace(mesh,TH)
pnp = Function(V,Fo+"accepted_solution.xml")

print "Number fo subspace", V.num_sub_spaces()

dofs_phi = V.sub(0).dofmap().dofs()
dofs_eta1 = V.sub(1).dofmap().dofs()
dofs_eta2 = V.sub(2).dofmap().dofs()

ar = pnp.vector().array()
for i in dofs_phi:
    ar[i]=ar[i]*1000/34.81356616
for i in dofs_eta1:
    ar[i]=np.exp(ar[i])*100.0
for i in dofs_eta2:
    ar[i]=np.exp(ar[i])*100.0

pnp.vector()[:] = ar

F1 = File(Fo+"accepted_solution2.pvd")
F1 << pnp


## NS
# Fo = "./DATA_250/"
# mesh = Mesh("mesh3.xml.gz")
# V2 = FunctionSpace(mesh, "RT", 1)
# u = Function(V2,Fo+"velocity_solution.xml")
# ar = u.vector().array()
# ar=ar*3.0*1E3
# u.vector()[:] = ar
# F1 = File(Fo+"velocity.pvd")
# F1 << u
