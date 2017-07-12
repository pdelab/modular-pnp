#! /usr/bin/python2.7

#from __future__ import print_function
from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


xc=0.0
yc=0.0
zc=0.0
rc=0.4

N = 100
dx = (1.2/2.0-rc)/float(N)
print dx
x = np.arange(N)*dx
p0=1.0
x0=0.0
Eps=1E-6
g = np.exp(x0)*( np.exp(p0/2.0) - 1.0 )/( np.exp(p0/2.0) + 1.0 )
K = np.sqrt(2.0/Eps)
print "K = ",K
pb=2*np.log( (1-g*np.exp(-x*K)) / (1+g*np.exp(-x*K)) )


plt.figure()
plt.plot(x, pb,'--')
plt.savefig("phib.png")
plt.close()