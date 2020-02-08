from Kmeans import *
import pandas as pd
from numpy import *
import numpy as np
from code import *
import math
import configparser
from decimal import Decimal
#读配置文件获取读数据地址和写结果地址
def parse_args(filename):
    cf = configparser.ConfigParser()
    cf.read(filename)
    kmeans = cf.items("kmeans")
    dic = {}
    for key, val in kmeans:
        dic[key] = val
    initpath = dic['initpath']
    clupath = dic['clusterpath']
    k = int(dic['k'])
    train = cf.items("train")
    dic1 = {}
    for key, val in train:
        dic1[key] = val
    path = dic1['train_dir']
    return clupath,initpath,path,k

clupath,initpath,path,k=parse_args('cfg.txt')#读配置文件获取获数据地址和写结果地址
print('read file:',initpath)
data = pd.read_csv(path,index_col=0,sep='\t').iloc[:,3:].dropna()
center=np.array(pd.read_csv(initpath,sep='\t',index_col=0))
factor=list(data.columns)
factor_num=len(factor)#变量个数
#标准化数据
data=np.array(data)
data=(data-np.mean(data,axis=0))/np.std(data,axis=0)
myCentroids, clustAssing, J, B = kmeans().kMeans(np.mat(data), k, np.array(center).reshape(k, -1))

classdic={}
for i in range(k):
    clslst=[]
    for idx,j in enumerate(np.array(clustAssing)):
        if i==j[0]:
            clslst.append(list(pd.read_csv(path,index_col=0,sep='\t').dropna().index)[i])
    classdic[str(i)]=clslst
for i in classdic:
    with open(clupath,'a') as c:
         c.write(i+':'+','.join(classdic[i])+'\n')