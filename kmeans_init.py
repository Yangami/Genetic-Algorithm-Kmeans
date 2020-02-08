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
    train = cf.items("kmeans")
    dic = {}
    for key, val in train:
        dic[key] = val
    initpath=dic['initpath']
    train = cf.items("train")
    dic1 = {}
    for key, val in train:
        dic1[key] = val
    clusterpath = dic1['train_dir']
    k=int(dic['k'])
    return clusterpath,initpath,k

def mybin(floatingPoint):
    return code().floatToBinary64(floatingPoint)
def mydec(binary):
    return code().binaryToFloat(binary)
def main(data,center,k):
    print(np.shape(center),np.shape(np.mat(data)),k)
    myCentroids, clustAssing,J,B = kmeans().kMeans(np.mat(data),k,np.array(center).reshape(k,-1))
    return B/np.sum(J)
def best(pop, fitvalue):  # 找出适应函数值中最大值，和对应的个体
    px = len(pop)
    bestindividual = []
    bestfit = fitvalue[0]
    for i in range(1, px):
        if (fitvalue[i] >= bestfit):
            bestfit = fitvalue[i]
            bestindividual = pop[i]
    return [bestindividual, bestfit]
def decodechrom(pop):  # 将种群的二进制基因转化为十进制（0,1023）
    temp = [];
    for i in range(len(pop)):
        t = 0;
        for j in range(10):
            t += pop[i][j] * (math.pow(2, j))
        temp.append(t)
    return temp

#将多个属性值连起来的浮点数编码化为属性值列表
def split_dec(code,code_len=64):
    temp=[]
    n=len(code)//64

    okl=code

    if n!=1:
        for l in range(n - 1):
            temp.append(mydec(okl[l *code_len:(l + 1) * code_len]))
        temp.append(mydec(okl[(n-1)*code_len:]))
    else:

        temp.apppend(mydec(okl))
    return temp
def calobjvalue(data,pop,factor_num,k):  # 计算目标函数值
    objvalue = [];
    for i in pop:
        objvalue.append(main(data, split_dec(i), k))

    return objvalue  # 目标函数值objvalue[m] 与个体基因 pop[m] 对应

def cumsum(fit_value):
    fit=[]
    for i in range(len(fit_value)-1):
        fit.append(np.sum(fit_value[:i+1]))
    fit.append(np.sum(fit_value))
    return fit
def calfitvalue(objvalue):  # 转化为适应值，目标函数值越大越好，负值淘汰。
    fitvalue = []
    temp = 0.0
    Cmin = 0;
    for i in range(len(objvalue)):
        if (objvalue[i] + Cmin > 0):
            temp = Cmin + objvalue[i]
        else:
            temp = 0.0
        fitvalue.append(temp)
    return fitvalue
def calfitvalue(objvalue):
    return objvalue
def sum(fitvalue):
    total = 0
    for i in range(len(fitvalue)):
        total += fitvalue[i]
    return total


def selection(pop, fitvalue):  # 自然选择（轮盘赌算法）
    newfitvalue = []
    totalfit = sum(fitvalue)
    for i in range(len(fitvalue)):
        newfitvalue.append(fitvalue[i] / totalfit)
    cumsum(newfitvalue)

    ms = [];
    ms.sort()
    poplen = len(pop)
    for i in range(poplen):
        ms.append(random.random())  # random float list ms
    ms.sort()
    fitin = 0
    newin = 0
    newpop = pop
    newfitvalue=cumsum(newfitvalue)
    while newin < poplen:
        if (ms[newin] < newfitvalue[fitin]):
            newpop[newin] = pop[fitin]
            newin = newin + 1
        else:
            fitin = fitin + 1
    pop = newpop
    return pop
def crossover(pop, pc):  # 个体间交叉，实现基因交换
    poplen = len(pop)
    for i in range(poplen - 1):
        if (random.random() < pc):
            cpoint = random.randint(0, len(pop[0]))
            temp1 = ''
            temp2 = ''
            temp1+=(pop[i][0: cpoint])
            temp1+=(pop[i + 1][cpoint: len(pop[i])])
            temp2+=(pop[i + 1][0: cpoint])
            temp2+=(pop[i][cpoint: len(pop[i])])
            pop[i] = temp1
            pop[i + 1] = temp2
    return pop
def mutation(pop, pm):  # 基因突变
    px = len(pop)
    py = len(pop[0])
    for i in range(px):
        if (random.random() < pm):
            mpoint = random.randint(0, py - 1)
            if (pop[i][mpoint] == 1):
                pop[i][mpoint] = 0
            elif(pop[i][mpoint] == 0):
                pop[i][mpoint] = 1
            else:
                pass
    return pop




#参数初始化
popsize = 50  # 种群的大小
clupath,initpath,k=parse_args('cfg.txt')#读配置文件获取获数据地址和写结果地址
print('read file:',clupath)
data = pd.read_csv(clupath,index_col=0,sep='\t').iloc[:,3:]
data=data.dropna()
factor=list(data.columns)
factor_num=len(factor)#变量个数
pc=0.5
pm=0.05
#计算各属性最小值和全距，以确定随机种群取值范围
fc_min,fc_d=[],[]
for i in data:
    fc_min.append(np.min(data[i]))
    fc_d.append(np.max(data[i])-np.min(data[i]))
#标准化数据
data=np.array(data)
data=(data-np.mean(data,axis=0))/np.std(data,axis=0)

results = []
bestfit = 0
fitvalue = []
tempop = []
# np.random.seed(21)
pop = []
for i in range(popsize):
    p = ''
    for _ in range(k):
        for j in range(factor_num):
            p += mybin(np.random.rand())
    pop.append(p)

for i in range(100):  # 繁殖100代
    #print(str(i)+
    objvalue = calobjvalue(data,pop,factor_num,k)  # 计算目标函数值
    print('obj',objvalue)
    fitvalue = calfitvalue(objvalue);  # 计算个体的适应值
    print('fit',fitvalue)
    [bestindividual, bestfit] = best(pop, fitvalue)  # 选出最好的个体和最好的函数值

    if len(bestindividual)!=0:
        results.append([bestfit, split_dec(bestindividual)])

        #print('exept',bestindividual)# 每次繁殖，将最好的结果记录下来
    pop=selection(pop, fitvalue)  # 自然选择，淘汰掉一部分适应性低的个体
    pop=crossover(pop, pc)  # 交叉繁殖
   # print('newpop',pop)
    pop=mutation(pop, pm)  # 基因突变
    print('pop',pop)
    print('result',bestfit,results)  # 打印函数最大值和对应的
kmeans_init=pd.DataFrame(np.array(split_dec(bestindividual)).reshape(k,-1))
kmeans_init.to_csv(initpath,sep='\t')