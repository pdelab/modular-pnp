#!/usr/bin/env python2.7

import numpy as np
import matplotlib.pyplot as plt

with open("eps=1/data.txt") as f:
    data = f.read()
data = data.split('\n')
nn=len(data)
s=5
t = [row.split('\t')[0] for row in data[s:nn-1]]
it = [row.split('\t')[1] for row in data[s:nn-1]]
res= [row.split('\t')[2] for row in data[s:nn-1]]
cation = [row.split('\t')[3] for row in data[s:nn-1]]
anion= [row.split('\t')[4] for row in data[s:nn-1]]
potential = [row.split('\t')[5] for row in data[s:nn-1]]
Energy = [row.split('\t')[6] for row in data[s:nn-1]]
TimeElaspsed =[row.split('\t')[7] for row in data[s:nn-1]]
MeshSize = [row.split('\t')[8] for row in data[s:nn-1]]

toto=6
grr=0.0
for i in range(toto+1):
    grr+=float(TimeElaspsed[i])
print t[toto],grr

plt.figure()
plt.plot(t,cation,'*',label="cation")
plt.plot(t,anion,'--',label="anion")
plt.plot(t,potential,label="potential")
plt.legend(loc='upper right')
plt.xlabel('time')
plt.ylabel('L2 norm of the derivative in time')
# plt.ylim(0,1.0)
# plt.xlim(0,10)
#title('About as simple as it gets, folks')
plt.savefig("time_diff.eps")
