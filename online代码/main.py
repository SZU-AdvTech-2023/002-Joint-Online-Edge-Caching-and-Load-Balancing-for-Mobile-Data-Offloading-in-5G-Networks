import numpy as np
from solve import ProblemDriver
import const as c
#

X_,Y_=[],[]
c.t=0

while c.t+c.H<c.T:
    var1,var2=ProblemDriver.run()
    var1 = np.array(var1).reshape(c.H, c.K)
    var2 = np.array(var2).reshape(c.K, c.nMU, c.H)
    X_.append(var1)
    Y_.append(var2)

    c.t+=1
X_=np.array(X_)
Y_=np.array(Y_)
"""
X_ : [T-H][H][K]
Y_ : [T-H][K][(nMU,H)]
"""
XOut,YOut=[],[]

for t in range(0,c.v):
    acc=[]
    for i in range(t):
        acc.append(X_[t-i][i][:])
    XOut.append(np.sum(acc, axis=0) / (t+1))
    acc=[]
    for i in range(t):
        acc.append(Y_[t-i,:,:,i])
    YOut.append(np.sum(acc, axis=0) / (t+1))

for t in range(c.v,c.T-c.H):
    acc=[]
    for i in range(c.v):
        acc.append(X_[t-i][i][:])
    XOut.append(np.sum(acc, axis=0) / c.v)
    acc=[]
    for i in range(c.v):
        acc.append(Y_[t-i,:,:,i])
    YOut.append(np.sum(acc, axis=0) / c.v)

XOut=XOut[1:]
YOut=YOut[1:]
print("c.H:",c.H,"c.v:",c.v,"cost",ProblemDriver.count_cost(XOut,YOut))
#print(XOut)
#print(YOut)
