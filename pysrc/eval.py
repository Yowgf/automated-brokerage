# Evaluation part of the automated house selling pipeline

from .preprocessing import *
from .cdm23 import *

def selectBestK(gaps, gapSigmas):
    # The rule is to select the first K that matches the condition
    # k* = argmin {gap(k) >= gap(k+1) - sigma_w(k+1)

    # gaps[0] correspond to K = 2
    
    # Initialize
    optimalK = 2

    if len(gaps) <= 1:
        return optimalK
    
    for i in range(len(gaps)):
        if gaps[i] >= gaps[i + 1] - gapSigmas[i + 1]:
            optimalK = i + 2
            break

    return optimalK
    
def getRepresentatives(df, maxK=None):
    normalize(df)
    
    # if K was not set
    if maxK == None:
        maxK = 3

    nIter = maxK - 1
    t = 5
    gaps, gapSigmas = computeGaps(df, nIter, t)

    optimalK = selectBestK(gaps, gapSigmas)
    
    # Do the fitting for the optimalK
    bestKMeansRun = runKmeans(df.values, optimalK)

    reps = bestKMeansRun.cluster_centers_
    
    return pd.DataFrame(reps, columns=df.columns)
