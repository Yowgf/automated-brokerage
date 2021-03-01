import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt

from pysrc.preprocessing import *
from pysrc.utils import *
df = loadPreprocessed()

def setMplParams():
    # Dont have time to learn seaborn
    # Will just use matplotlib
    plt.rcParams["figure.figsize"]     = (12, 8)
    plt.rcParams["axes.labelsize"]     = 11
    plt.rcParams["axes.labelweight"]   = "bold"
    plt.rcParams["scatter.edgecolors"] = "white"
    plt.rcParams["lines.linewidth"]    = 0.5

def plotLivPrice():    
    plt.xlabel("sqft_living")
    plt.ylabel("price - million US$")
    plt.scatter(df.sqft_living.values, df.price.values)
    curLocs, _ = plt.yticks()
    plt.yticks(curLocs[1:-1], [str(int(price)) for price in curLocs[1:-1] / 1e6], color='k')

    plt.savefig("doc/history/cdm-23/sqft_living-price.png")
    plt.show()

def plotLiv15Price():
    plt.xlabel("sqft_living15")
    plt.ylabel("price - million US$")
    plt.scatter(df.sqft_living15.values, df.price.values)
    curLocs, _ = plt.yticks()
    plt.yticks(curLocs[1:-1], [str(int(price)) for price in curLocs[1:-1] / 1e6])

    plt.savefig("doc/history/cdm-23/sqft_living15-price.png")
    plt.show()

def showWaterFront():
    withoutWaterDf = df[df.waterfront == 0].price
    withoutWaterMean = withoutWaterDf.mean()
    withoutWaterStd = withoutWaterDf.std()
    withoutWaterRange = np.array([withoutWaterMean - withoutWaterStd, withoutWaterMean + withoutWaterStd], dtype=int)
    
    withWaterDf = df[df.waterfront == 1].price
    withWaterMean = withWaterDf.mean()
    withWaterStd = withWaterDf.std()
    withWaterRange = np.array([withWaterMean - withWaterStd, withWaterMean + withWaterStd], dtype=int)
    
    print(f"Price without waterfront: {withoutWaterRange}")
    print(f"Price with waterfront: {withWaterRange}")

def plotScatterMatrix(df):
    hist_kwds = {"hist_kwds": {"edgecolor": "k", "bins": 30}}
    scatter_kwds = {"linewidth": 0.2, "edgecolor": "white"}
    scatter_matrix(df.iloc[::, :4], figsize=(13, 8), **hist_kwds, **scatter_kwds)
    plt.savefig("doc/history/cdm-23/scatter-matrix.png")
    plt.show()

def chooseBiggestSigmas():
    sigmas = np.linalg.svd(df, compute_uv=False)
    chosenSigmas = list()
    head = 0
    while sum(chosenSigmas) / sigmas.sum() < 0.95:
        chosenSigmas.append(sigmas[head])
        head += 1

    return np.array(chosenSigmas)

def buildReDf(chosenSigmas):
    u, s, vh = np.linalg.svd(df, full_matrices=False)
    dim = len(chosenSigmas)
    aproxDf = u[::, :dim] @ np.diag(s[:dim])
    reDf = pd.DataFrame(aproxDf, columns=["pc" + str(i) for i in range(dim)])

    return reDf

def plotReDf(reDf):
    # Plot the reduced dimension data
    plt.scatter(reDf.values[::, 0], reDf.values[::, 1], edgecolor="white", linewidth=0.5)
    plt.xlabel("pc1", size=11, weight="bold")
    plt.ylabel("pc2", size=11, weight="bold")
    plt.savefig("doc/history/cdm-23/data2d.png")
    plt.show()

# 2. I have to generate samples of the dataset following the
#    Monte Carlo sampling methodology.
def generateSample(df, sampSize):
    minMaxs = (df.min(), df.max())
    randSamp = np.zeros((sampSize, df.shape[1]))
    for i in range(randSamp.shape[0]):
        for j in range(randSamp.shape[1]):
            randSamp[i, j] = np.random.rand() * (minMaxs[1][j] - minMaxs[0][j]) + minMaxs[0][j]

    return np.nan_to_num(randSamp)

# 3. I then run K-means on each sample like 100 times, to find
#    out the best clustering for that sample.
from sklearn.cluster import KMeans

def vecEucDist(vec1, vec2):
    return np.sqrt(((vec1 - vec2) ** 2).sum(axis=1))

def runKmeans(dataMatrix, K):
    dataMatrix = np.nan_to_num(dataMatrix)
    
    numberOfRuns = 10
    minError = np.inf
    minModel = None
    for run in range(numberOfRuns):
        model = KMeans(n_clusters=K, random_state=None)
        model.fit(dataMatrix)
        
        # Need to be able to measure clustering error
        error = model.inertia_ # sqrError(dataMatrix, model)
        
        if error < minError:
            minError = error
            minModel = model
    
    return minModel
    
# 4. Build the W similarity matrix for each cluster (1..K).
#    Calculate the W_in coefficient, which is just the sum of
#    all the values of that matrix. (should end up with like 3 
#    nested loops for this part)
def mapClusterDict(dataMatrix, labels):
    clusters = dict()
    for cluster in np.unique(labels):
        clusters[cluster] = list()
        
    for i in range(labels.size):
        clusters[labels[i]].append(dataMatrix[i])
    
    return clusters

def buildW(cPoints):
    cSize = cPoints.shape[0]         # Cluster size
    W = np.zeros((cSize, cSize))     # W has size cSize X cSize
    for i in range(cSize):           # Go through all the points
        W[i, i:] = vecEucDist(cPoints[i], cPoints[i:])
    
    return W

def buildW_in(clusters, k):
    cPoints = np.array(clusters[k]) # Cluster points
    W_in = buildW(cPoints)
    
    return W_in
    
def calcWeightIn(dataMatrix, KMeansModel):
    centroids = KMeansModel.cluster_centers_
    labels = KMeansModel.labels_
    
    if centroids.size == 0 or labels.size == 0:
        raise AttributeError("Invalid empty model")
    
    # Build association cluster -> its points
    clusters = mapClusterDict(dataMatrix, labels)
    
    # Build W_in for each cluster
    numClusters = len(clusters.keys())
    Ws = np.zeros(numClusters)
    for k in range(numClusters):
        W_in = buildW_in(clusters, k)
        
        # W_in only has upper right entries non-null, so
        # we dont have to divide them by two
        Ws[k] = W_in.sum()
    
    return Ws

def calcAllSampleWeights(df, sampSize, K, t):
    W_ins = np.zeros(t)
    for sampIdx in range(t):
        randSamp = generateSample(df, sampSize)
        bestKMeansRun = runKmeans(randSamp, K)

        W_ins[sampIdx] = calcWeightIn(randSamp, bestKMeansRun).sum()
    
    return W_ins

def computeGapStatistic(df, sampSize, K, t):
    W_ins = calcAllSampleWeights(df, sampSize, K, t)

    # 5. Get the mean "mu" and standard deviation of the W_in's
    # use the logarithm
    mu = np.log(W_ins).mean()
    sigma = np.sqrt(((np.log(W_ins) - mu) ** 2).mean())

    # 6. The gap statistic for K is then given as
    #    gap(K) = mu - log(W_in(original dataset))
    # Have to calculate the W_in of original dataset
    bestKMeansRun = runKmeans(df.values, K)
    expectW_in = calcWeightIn(df.values, bestKMeansRun).sum()
    
    return mu - np.log(expectW_in), sigma

# Computes the whole vector of gaps, for n iterations
def computeGaps(df, nIter, t=5):
    gaps = np.zeros(nIter)
    gapSigmas = np.zeros(nIter)
    sampSize = df.shape[0]
    for K in range(2, nIter + 2):
        gaps[K - 2], gapSigmas[K - 2] = computeGapStatistic(df, sampSize, K, t)
        print("\nCalculated gap statistic: ", gaps[K - 2])

    return gaps, gapSigmas

def plotClustering2d(reDf, optimalK):
    # Plot each cluster separately
    bestKMeansRun = runKmeans(reDf.values, optimalK)
    clusters = mapClusterDict(reDf.values, bestKMeansRun.labels_)
    for k in clusters.keys():
        cPoints = np.array(clusters[k])
        plt.scatter(cPoints[::, 0], cPoints[::, 1], edgecolor="white", linewidth=0.5)

    # Plot the centroids above
    centroids = np.array(bestKMeansRun.cluster_centers_)
    plt.scatter(centroids[::, 0], centroids[::, 1], color='k', label="centroids")

    plt.xlabel("pc1", size=11, weight="bold")
    plt.ylabel("pc2", size=11, weight="bold")
    plt.legend(labels=["cluster 1", "cluster 2", "centroids"])
    plt.savefig("doc/history/cdm-23/clustering2d.png")
    plt.show()

def plotGapsDist(gaps, gapSigmas): 
    x, y = np.arange(2, len(gaps) + 2), gaps
    plt.figure(1, figsize=(12, 8))
    plt.errorbar(x, y, linewidth=1, xerr=None, yerr=2 * gapSigmas, ecolor="grey", elinewidth=4)
    plt.xlabel("K", weight="bold", size=11)
    plt.ylabel("gap", weight="bold", size=11)
    plt.savefig("doc/history/cdm-23/gaps.png")
    plt.show()
