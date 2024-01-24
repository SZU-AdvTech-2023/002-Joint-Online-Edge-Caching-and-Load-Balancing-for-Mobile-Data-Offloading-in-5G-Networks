import const as c
import numpy as np
from solve import P1,P2,Problem

LB=-1e10
UB=1e10
l_max=1000
e=0.0001

X=None
Y=None
l_=1
while  (UB-LB)/UB>e and l_<=l_max:
    X,cost1=P1.opt()
    Y,cost2=P2.opt()
    if cost1+cost2>LB:
        LB=cost1+cost2

    UB=Problem.objective_function(X,Y)
    
    Problem.update_mu(X,Y,l_)

    #print("mu",c.mu)
    print("step:",l_,"(UB-LB)/UB=",(UB-LB)/UB)
    l_+=1

print("cost=",UB)
