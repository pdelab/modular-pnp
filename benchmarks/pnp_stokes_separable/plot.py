#! /usr/bin/python2.7

#from __future__ import print_function
from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


# Fo = "./meshOut_2/"
Fo = "./output/"

# Set backend to PETSC
# parameters['linear_algebra_backend'] = 'Eigen'
# parameters["linear_algebra_backend"] = "PETSc"
parameters["allow_extrapolation"] = True


# Load mesh and define function space
mesh = Mesh("mesh.xml.gz");
V = FunctionSpace(mesh, "CG", 1)
phi = Function(V,Fo+"potential.xml")

xc=[-2.36111111111,2.5,3.47222222222]
yc=[-2.22222222222,1.52777777778,-2.08333333333]
zc=[0.972222222222,-2.63888888889,-2.08333333333]
rc=[1.25,1.25,1.25]

x = np.arange(50)*0.01
p0=1.0
x0=0.0
Eps=1E0
g = exp(x0)*( exp(p0/2.0) - 1 )/( exp(p0/2.0) + 1 )
pb=2*np.log( (1-g*np.exp(-x/np.sqrt(Eps))) / (1+g*np.exp(-x/np.sqrt(Eps))) )


vphi_t = np.zeros(1);
vphi1 = np.zeros(50);
vphi2 = np.zeros(50);
vphi3 = np.zeros(50);
vphi4 = np.zeros(50);
x_coord1=np.zeros(3);
x_coord2=np.zeros(3);
x_coord3=np.zeros(3);
x_coord4=np.zeros(3);
x_coord1[0]=xc[0]; x_coord1[1]=yc[0]; x_coord1[2]=zc[0]+1.25;
x_coord2[0]=xc[0]; x_coord2[1]=yc[0]-1.25; x_coord2[2]=zc[0];
x_coord3[0]=xc[1]; x_coord3[1]=yc[1]; x_coord3[2]=zc[1]+1.25;
x_coord4[0]=xc[1]; x_coord4[1]=yc[1]-1.25; x_coord4[2]=zc[1];
for j in range(50):
    phi.eval(vphi_t,x_coord1);
    vphi1[j]=vphi_t[0]
    phi.eval(vphi_t,x_coord2);
    vphi2[j]=vphi_t[0]
    phi.eval(vphi_t,x_coord3);
    vphi3[j]=vphi_t[0]
    phi.eval(vphi_t,x_coord4);
    vphi4[j]=vphi_t[0]
    x_coord1[2]+=0.01;
    x_coord2[1]-=0.01;
    x_coord3[2]+=0.01;
    x_coord4[1]-=0.01;



plt.figure()
plt.plot(x, vphi1)
plt.plot(x, vphi2,'*')
plt.plot(x, vphi3)
plt.plot(x, vphi4,'--')
plt.plot(x, pb,'--')
plt.legend(('PNP','PNP','PNP','PNP','PB'),loc=4)
plt.savefig("potential_cut.eps")
plt.close()
