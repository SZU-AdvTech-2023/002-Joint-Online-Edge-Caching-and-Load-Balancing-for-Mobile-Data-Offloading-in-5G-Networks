from zipf import GetZipfDemand
import numpy as np
import random

K=3
T=100
nMU=3
C=1
B=3
BETA=100

H=10
v=5
t=0

random.seed(9743)
requestPerNode=random.randint(1,4)#30个MU加起来的请求是[30,120]？
mu=0.1*np.array([np.ones((nMU, H)) for x in range(K)])
l=GetZipfDemand(nMU, K, T, requestPerNode, [30, 0.8, 0.05])
"""
l:Array of numpy matrix, dimension: [[K][(nMU,T)]]
"""
