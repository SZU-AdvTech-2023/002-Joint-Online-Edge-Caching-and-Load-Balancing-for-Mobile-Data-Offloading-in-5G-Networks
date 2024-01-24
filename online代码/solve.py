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
        X=X.reshape(c.H, c.K)

        cost=0
        for t in range(c.H - 1):
            xdif=X[t+1]-X[t]
            x=[max(xdif[k],0) for k in range(c.K)]#H里的d
            cost+=c.BETA*sum(x)

        for t in range(c.H):
            for m in range(c.nMU):
                for k in range(c.K):
                    cost-=X[t][k]*c.mu[k][(m,t)]

        return cost

    @staticmethod
    def constraint(X):#(1)
        X=X.reshape(c.H, c.K)
        return [c.C - sum(X[t]) for t in range(c.H)]

    @staticmethod
    def opt() -> ([],int):#返回一维list，在函数外传递的XY都是一维list
        constraints=[
            {'type': 'ineq', 'fun': P1.constraint}
            ]
        bounds = [(0, 1) for i in range(c.H * c.K)]#(10)
        result = minimize(P1.objective_function, 0.1 * np.ones(c.H * c.K), method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x, result.fun

class P2:
    @staticmethod
    def objective_function(Y):
        Y=Y.reshape(c.K, c.nMU, c.H)

        cost=0
        for t in range(c.H):
            ft=0
            for m in range(c.nMU):
                acc=0
                for k in range(c.K):
                    acc += (1 - Y[k][(m, t)]) * c.l[k][(m, t+c.t)]
                ft+=random.random()*acc
            cost+=ft*ft
        for t in range(c.H):
            for m in range(c.nMU):
                for k in range(c.K):
                    cost+=c.mu[k][(m,t)]*Y[k][(m,t)]
        return cost

    @staticmethod
    def constraint(Y):#(2)
        Y=Y.reshape(c.K, c.nMU, c.H)
        r=[]
        for t in range(c.H):
            for m in range(c.nMU):
                for k in range(c.K):
                    r.append(c.B - Y[k][(m, t)] * c.l[k][(m, t+c.t)])
        return r

    @staticmethod
    def opt() -> ([],int):#返回一维list，在函数外传递的XY都是一维list
        random.seed(9744)
        constraints=[
            {'type': 'ineq', 'fun': P2.constraint}
            ]
        bounds = [(0, 1) for i in range(c.H * c.K * c.nMU)]#(11)
        result = minimize(P2.objective_function, 0.1 * np.ones(c.H * c.K * c.nMU), method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x, result.fun

class Problem:
    @staticmethod
    def objective_function(X,Y):
        X=np.array(X).reshape(c.H, c.K)
        Y=np.array(Y).reshape(c.K, c.nMU, c.H)
        random.seed(9743)

        cost=0
        for t in range(c.H - 1):
            xdif=X[t+1]-X[t]
            x=[max(xdif[k],0) for k in range(c.K)]#H里的d
            cost+=c.BETA*sum(x)

        for t in range(c.H):
            ft=0
            for m in range(c.nMU):
                acc=0
                for k in range(c.K):
                    acc += (1 - Y[k][(m, t)]) * c.l[k][(m, t+c.t)]
                ft+=random.random()*acc
            cost+=ft*ft
        return cost

    @staticmethod
    def update_mu(X,Y,l_):
        X=np.array(X).reshape(c.H, c.K)
        Y=np.array(Y).reshape(c.K, c.nMU, c.H)

        for t in range(c.H):
            for m in range(c.nMU):
                for k in range(c.K):
                    g=Y[k][(m,t)]-X[t][k]
                    c.mu[k][(m,t)]=max(0 , c.mu[k][(m,t)]+g/(1+0.1*l_))

    @staticmethod
    def init_mu():
        c.mu = 0.1 * np.array([np.ones((c.nMU, c.H)) for x in range(c.K)])

class ProblemDriver:
    @staticmethod
    def run():
        LB=-1e10
        UB=1e10
        l_max=1000
        e=0.0001

        Problem.init_mu()
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
            #print("step:",l_,"(UB-LB)/UB=",(UB-LB)/UB)
            l_+=1

        return X,Y

    @staticmethod
    def count_cost(X,Y):
        X=np.array(X).reshape(c.T-c.H-1,c.K)
        Y=np.array(Y).reshape(c.T-c.H-1,c.K,c.nMU,)
        random.seed(9743)

        cost=0
        for t in range(c.T-c.H-1 - 1):
            xdif=X[t+1]-X[t]
            x=[max(xdif[k],0) for k in range(c.K)]#H里的d
            cost+=c.BETA*sum(x)
            
        for t in range(c.T-c.H-1):
            ft=0
            for m in range(c.nMU):
                acc=0
                for k in range(c.K):
                    acc+=(1-Y[t][k][m])*c.l[k][(m,t)]
                ft+=random.random()*acc
            cost+=ft*ft
        return cost
