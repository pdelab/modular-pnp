#! /usr/bin/python2.7

#from __future__ import print_function
from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


# Fo = "./meshOut_2/"
Fo = "./XML/"

# Set backend to PETSC
# parameters['linear_algebra_backend'] = 'Eigen'
# parameters["linear_algebra_backend"] = "PETSc"
parameters["allow_extrapolation"] = True

# Define boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return (abs(x[0] + 5.0)< 5.0*DOLFIN_EPS or abs(x[0] - 5.0) < 5.0*DOLFIN_EPS) and on_boundary

class DirichletBoundary2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

# Load mesh and define function space
mesh = Mesh(Fo+"mesh.xml");
CG = FiniteElement("CG", mesh.ufl_cell(), 1)
RT = FiniteElement("RT", mesh.ufl_cell(), 1)
DG = FiniteElement("DG", mesh.ufl_cell(), 0)
V = FunctionSpace(mesh,  RT * DG )


V_CG = FunctionSpace(mesh,  CG)
V_RT = FunctionSpace(mesh,  RT)
V_DG = FunctionSpace(mesh,  DG)
AnAn = Function(V_CG,Fo+"anion.xml")
CatCat = Function(V_CG,Fo+"cation.xml")
EsEs= Function(V_CG,Fo+"potential.xml")
uu = Function(V_RT,Fo+"velocity.xml")
pp = Function(V_DG,Fo+"pressure.xml")

alpha1 = 1E-6
alpha2 = 1E-12
mu = 0.1
qp=1.0
qn=-1.0
eps = 1.0
n_vec = FacetNormal( mesh )
h = CellSize( mesh )
h_avg = ( h('+')+h('-') )/2.0
Dp = 1.0
Dn = 1.0
eps=0.1


phi = TestFunction(V_CG)
L1 =   eps * inner(grad(EsEs),grad(phi))*dx \
     - (qp*exp(CatCat) + qn*exp(AnAn))*phi*dx


cat = TestFunction(V_CG)
L2  = - ( Dp*exp(CatCat)* (inner(grad(CatCat),grad(cat)) + qp*inner(grad(EsEs),grad(cat))) )*dx \
 + ( exp(CatCat)*(inner(uu,grad(cat))) )*dx

an = TestFunction(V_CG)
L3 = - ( Dn*exp(AnAn  )* (inner(grad(AnAn  ),grad(an )) + qn*inner(grad(EsEs), grad(an))) )*dx \
  + ( exp(AnAn)*(inner(uu,grad(an)))  )*dx


v = TestFunction(V_RT)
L4 = - ( 2.0*mu* inner( sym(grad(uu)), sym(grad(v)) ) )*dx   +   ( pp*div(v) )*dx   \
    - eps*inner( outer(grad(EsEs),grad(EsEs)) , grad(v) )*dx

q = TestFunction(V_DG)
L5 = -   ( div(uu)*q )*dx

v = TestFunction(V_RT)
L6 = - eps*inner( outer(grad(EsEs),grad(EsEs)) , grad(v) )*dx \
    + eps/2.0*inner(grad(EsEs),grad(EsEs))*div(v)*dx \
    + ( (qp*exp(CatCat)+qn*exp(AnAn))*inner(grad(EsEs),v) )*dx

for i in range(len(pp.vector()[:])):
    if pp.vector()[i]==0:
        print i

u0 = Constant((0.0))
bc = DirichletBC(V_CG, u0, DirichletBoundary())
u3 = Constant((0.0,0.0,0.0))
bc2 = DirichletBC(V_RT, u3, DirichletBoundary())


b1 = assemble(L1)
bc.apply(b1)

b2 = assemble(L2)
bc.apply(b2)

b3 = assemble(L3)
bc.apply(b3)

b4 = assemble(L4)
bc2.apply(b4)

b5 = assemble(L5)
b5[1202]=0.0;

b6 = assemble(L6)
bc2.apply(b6)

print "norm phi = ", b1.norm("l2")
print "norm Cat = ", b2.norm("l2")
print "norm An = ", b3.norm("l2")
print "norm u = ", b4.norm("l2")
print "norm div = ", b5.norm("l2")
print "norm diff = ", b6.norm("l2")
