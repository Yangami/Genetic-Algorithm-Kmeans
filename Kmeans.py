# encoding=utf-8
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from math import *
import pandas as pd
from numpy import *
data=pd.read_csv('./data/fund_2016q1.txt')
stocks=data.index
data=np.array(data.iloc[:,1:])
data=(data-np.mean(data,axis=0))/np.std(data,axis=0)
data=data.tolist()
txtarr=data
class kmeans:
    def __init__(self):
        self=self
# coding=utf-8
    # 计算两个向量的距离，用的是欧几里得距离
    def distEclud(self,vecA, vecB):
        return sqrt(sum(power(vecA - vecB, 2)))
    # 随机生成初始的质心（初始方式是随机选K个点）
    def randCent(self,dataSet, k):
        n = shape(dataSet)[1]
        centroids = mat(zeros((k, n)))
        for j in range(n):
            minJ = min(dataSet[:, j])
            rangeJ = float(max(array(dataSet)[:, j]) - minJ)
            centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
        return centroids
    def kMeans(self,dataSet, k,pop, distMeas=distEclud, createCent=randCent):
        for i in dataSet:
            if np.nan in i:
                print('false1')
        print(len(dataSet),dataSet[0])
      #  print('aaa',dataSet,np.shape(dataSet))
        m = shape(dataSet)[0]
        clusterAssment = mat(zeros((m, 2)))  # create mat to assign data points
        # to a centroid, also holds SE of each point
        centroids=pop
        clusterChanged = True

        while clusterChanged:
            clusterChanged = False
            for i in range(m):  # for each data point assign it to the closest centroid
                minDist = inf
                minIndex = -1
                for j in range(k):
                    distJI = distMeas(self,centroids[j, :], dataSet[i, :])
                    if distJI < minDist:
                        minDist = distJI;
                        minIndex = j
                if clusterAssment[i, 0] != minIndex:
                    clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2
            for cent in range(k):  # recalculate centroids
                ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
                centroids[cent, :] = mean(ptsInClust, axis=0)  # assign centroid to mean
        J=[]
        center=mean(np.array(centroids),axis=0)
       # print('sssssss',center,[distMeas(self,center,i) for i in centroids])
        B=np.sum([distMeas(self,center,i) for i in centroids])
        for cent in range(k):  # recalculate centroids
            jk=0
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            for point in ptsInClust:
                jk+=distMeas(self,point,centroids[cent])
            J.append(jk)
        return centroids, clusterAssment,J,B
def main():
    dataMat = mat(data)
    myCentroids, clustAssing,J,B = kmeans('_','_').kMeans(dataMat, 4)
    print(myCentroids,clustAssing,np.sum(J),B)
    return B/np.sum(J)

#print(firclu[0],[list(firclu[0]).count(i) for i in set(firclu[0])],set(firclu[0]))
