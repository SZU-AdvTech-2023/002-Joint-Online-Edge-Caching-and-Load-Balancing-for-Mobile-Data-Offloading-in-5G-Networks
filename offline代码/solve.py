import const as c
import random
from scipy.optimize import minimize
import numpy as np

class P1:

    @staticmethod
    def objective_function(X):
        """
        X.shape=(c.T*c.K)
        """
        X=X.reshape(c.T,c.K)
        
        cost=0
        for t in range(c.T-1):
            xdif=X[t+1]-X[t]
            x=[max(xdif[k],0) for k in range(c.K)]#H里的d
            cost+=c.BETA*sum(x)
            
        for t in range(c.T):
            for m in range(c.nMU):
                for k in range(c.K):
                    cost-=X[t][k]*c.mu[k][(m,t)]
                    
        return cost
    
    @staticmethod
    def constraint(X):#(1)
        X=X.reshape(c.T,c.K)
        return [c.C-sum(X[t])for t in range(c.T)]
    
    @staticmethod
    def opt():
        constraints=[
            {'type': 'ineq', 'fun': P1.constraint}
            ]
        bounds = [(0, 1)for i in range(c.T*c.K)]#(10)
        result = minimize(P1.objective_function, 0.33*np.ones(c.T*c.K), method='COBYLA', bounds=bounds, constraints=constraints) 
        return result.x, result.fun

class P2:
    @staticmethod
    def objective_function(Y):
        Y=Y.reshape(c.K,c.nMU,c.T)
        
        cost=0
        for t in range(c.T):
            ft=0
            for m in range(c.nMU):
                acc=0
                for k in range(c.K):
                    acc+=(1-Y[k][(m,t)])*c.l[k][(m,t)]
                ft+=random.random()*acc
                #ft+=c.wmn[m]*acc
            cost+=ft*ft
        for t in range(c.T):
            for m in range(c.nMU):
                for k in range(c.K):
                    cost+=c.mu[k][(m,t)]*Y[k][(m,t)]
        return cost
    
    @staticmethod
    def constraint(Y):#(2)
        Y=Y.reshape(c.K,c.nMU,c.T)
        r=[]
        for t in range(c.T):
            r.append(c.B-np.sum(np.multiply(Y[:,:,t],np.array(c.l)[:,:,t])))
            
        return r

    @staticmethod
    def opt():
        constraints=[
            {'type': 'ineq', 'fun': P2.constraint}
            ]
        bounds = [(0, 1)for i in range(c.T*c.K*c.nMU)]#(11)
        result = minimize(P2.objective_function, 0.1*np.ones(c.T*c.K*c.nMU), method='COBYLA', bounds=bounds, constraints=constraints)
        return result.x, result.fun
    
class Problem:
    @staticmethod
    def objective_function(X,Y):
        X=np.array(X).reshape(c.T,c.K)
        Y=np.array(Y).reshape(c.K,c.nMU,c.T)
        random.seed(9743)

        cost=0
        for t in range(c.T-1):
            xdif=X[t+1]-X[t]
            x=[max(xdif[k],0) for k in range(c.K)]#H里的d
            cost+=c.BETA*sum(x)
            
        for t in range(c.T):
            ft=0
            for m in range(c.nMU):
                acc=0
                for k in range(c.K):
                    acc+=(1-Y[k][(m,t)])*c.l[k][(m,t)]
                ft+=random.random()*acc
                #ft+=c.wmn[m]*acc
            cost+=ft*ft
        return cost
    
    @staticmethod
    def update_mu(X,Y,l_):
        X=np.array(X).reshape(c.T,c.K)
        Y=np.array(Y).reshape(c.K,c.nMU,c.T)
        
        for t in range(c.T):
            for m in range(c.nMU):
                for k in range(c.K):
                    g=Y[k][(m,t)]-X[t][k]
                    c.mu[k][(m,t)]=max(0 , c.mu[k][(m,t)]+g/(1+0.1*l_))
