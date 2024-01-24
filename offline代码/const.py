from zipf import GetZipfDemand
import numpy as np
import random

K=3
T=20
nMU=6
C=1
B=3
BETA=200

random.seed(9743)
requestPerNode=random.randint(1,4)#30个MU加起来的请求是[0,100]？
l=GetZipfDemand(nMU,K,T,requestPerNode,[30,0.8,0.00])
wmn=[random.random() for i in range(nMU)]

"""
l:Array of numpy matrix, dimension: [[K][(nMU,T)]]
"""


"""
K=30
T=100
nMU=1
C=5
B=30
BETA=100
omg=0.5#random.random()
requestPerNode=random.ranint(100)#30个MU加起来的请求是[0,100]？
l=GetZipfDemand(nMU,K,T,requestPerNode,(30,0.8,0))
"""

mu=0.1*np.array([np.ones((nMU, T)) for x in range(K)])
