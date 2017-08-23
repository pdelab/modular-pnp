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
