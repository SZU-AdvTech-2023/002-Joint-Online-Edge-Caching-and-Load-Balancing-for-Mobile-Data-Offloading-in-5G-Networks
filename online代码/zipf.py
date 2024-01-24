import random as rd
import numpy as np
#import settings as st


def NoisedZipf(q: float, alpha: float, itemNo: int, noiseRatio: float = 0.05):
    """Noised Zipf: Generate zipf distribution and add noise
    :returns: The probability (p) and cumulative (cum) distributions
    :param q: q
    :param alpha: alpha
    :param itemNo: Number of items
    :param noiseRatio: The ratio of noise. In fractions (e.g., 0.1 for 10%)
    """
    #definition and processing
    p = []
    cum = []
    n = itemNo
    # calculate k
    K = 0
    for i in range(1, n+1):
        K += 1/((i+q)**alpha)
    K = 1 / K
    # get sequence s
    for i in range(1, n+1):
        r = K / ((i + q) ** alpha)
        p.append(r)
    # add noise and normalize
    for i in range(n):
        # e.g., 0.95 + [0, 0.1) = [0.95, 1.05)
        p[i] = p[i] * ((1 - noiseRatio) + 2 * rd.random() * noiseRatio)
    s_p = sum(p)
    p[:] = [p_i / s_p for p_i in p]
    #p[:] = [p_i * st.n_n * 10 / s_p for p_i in p]
    # get cumulative distribution
    for i in range(n):
        cum.append(p[i])
        if i > 0:
            cum[i] += cum[i-1]

    # return
    return p, cum


def GetZipfDemand(nodesNo: int, fileNo: int, duration: int, requestPerNode: int, zipfParam: list = None):
    '''Get demands from noised zipf
    :returns: Array of numpy matrix, dimension: [[k][(m,t)]] 
    :param nodesNo: Number of nodes (n) #MU=30
    :param fileNo: Number of files (k)
    :param duration: Time durations (t)
    :param requestPerNode: At any time each node will request how many file
    :param zipfParam: Parameters for noised zipf [q, alpha, noiseRatio(=0.05)]
    '''
    # initialize
    rd.seed(9743)
    l = [np.zeros((nodesNo, duration)) for x in range(fileNo)]
    # zipf parameters
    q = 30
    alpha = 0.8
    itemNo = fileNo
    noiseRatio = 0.05
    if (zipfParam != None):
        q = zipfParam[0]
        alpha = zipfParam[1]
        noiseRatio = zipfParam[2]
    # sample
    for m in range(nodesNo):
        for t in range(duration):
            cn = NoisedZipf(q,alpha,itemNo,noiseRatio)[1]
            rn = requestPerNode
            while rn > 0:
                choice = rd.random()
                for i in range(len(cn)):
                    if choice < cn[i]:
                        l[i][(m, t)] += 1
                        break
                rn -= 1
    # return
    return l
