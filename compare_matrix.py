#!/usr/bin/env python2.7

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla

with open("21.txt") as f:
    data = f.read()
data = data.split('\n')
nn=len(data)
# t = [row.split('\t')[0] for row in data[1:]]
row = data[0].split("  ")
N1 = int(row[0])
N2 = int(row[1])
A11 = np.zeros((N1,N2),'d')
print N1, N2
for row in data[1:nn-1]:
    _row = row.split("  ")
    # print _row
    i = int(_row[0])
    j = int(_row[1])
    A11[i,j] = float(_row[2])

with open("21b.txt") as f:
    data = f.read()
data = data.split('\n')
nn=len(data)
# t = [row.split('\t')[0] for row in data[1:]]
row = data[0].split("  ")
N1 = int(row[0])
N2 = int(row[1])
A11b = np.zeros((N1,N2),'d')
print N1, N2
for row in data[1:nn-1]:
    _row = row.split("  ")
    # print _row
    i = int(_row[0])
    j = int(_row[1])
    A11b[i,j] = float(_row[2])

# E11=np.linalg.eig(A11b)[0]
# E11b=np.linalg.eig(A11b)[0]
# f1=open('Eig.txt', 'w+')
# print >> f1, sorted(E11)
# f1=open('Eigb.txt', 'w+')
# print >> f1, sorted(E11b)
# print sum([1 for i,j in zip(E11,E11b) if i==j])

E11=np.linalg.svd(A11b)[0]
E11b=np.linalg.svd(A11b)[0]
f1=open('Eig.txt', 'w+')
print >> f1, (E11)
f1=open('Eigb.txt', 'w+')
print >> f1, (E11b)
