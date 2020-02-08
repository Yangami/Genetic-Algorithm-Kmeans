from Kmeans import *
import pandas as pd
from numpy import *
import numpy as np
from code import *
import configparser
import random
import math
#读配置文件获取读数据地址和写结果地址
def parse_args(filename):
    cf = configparser.ConfigParser()
    cf.read(filename)
    train = cf.items("train")
    dic = {}
    for key, val in train:
        dic[key] = val
    traindir = dic['train_dir']
    bafdir=dic['baf_dir']
    return traindir,bafdir
#目标函数。
#输入 各属性取值限制值、df格式的数据、属性名列表
#输出目标函数值
def main_revenue(baffle, df,factor):
    pool = []
    for st in df.index:
        flag = 1
        for j,i in enumerate(factor):
            if df[i][st] < baffle[j]:
                flag=0
        if flag==1:
            pool.append(st)
    if len(pool)==0:
        return -999
    else:
        return np.sum([df['r'][st] for st in pool])/len(pool)
#调用code.py对单个读点数进行编码
def mybin(floatingPoint):
    return code().floatToBinary64(floatingPoint)
#浮点数解码
def mydec(binary):
    return code().binaryToFloat(binary)
# 找出适应函数值中最大值，和对应的个体。以最终best作为最终模型的结果
def best(pop, obj):
    px = len(pop)
    bestindividual = pop[0]
    bestobj = obj[0]
    for i in range(1, px):
        if (obj[i] >= bestfit):
            bestobj = obj[i]
            bestindividual = pop[i]

    return (bestindividual, bestobj)

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
 #计算目标函数值
#输入浮点数编码pop、df数据、属性名称列表
def calobjvalue(pop,data,factor):

    df=data
    objvalue=[]
    for j,i in enumerate(pop):
        temp1=split_dec(i)
        objvalue.append(main_revenue(temp1,df,factor))

    return objvalue

def cumsum(fit_value):
    fit=[]
    for i in range(len(fit_value)-1):
        fit.append(np.sum(fit_value[:i+1]))
    fit.append(np.sum(fit_value))
    return fit
# 目标函数值转化为适应值，目标函数值越大越好，负值淘汰
def calfitvalue(objvalue):
    fitvalue = []
    h1 = 0.001
    cmax = np.max(objvalue)

    for i in objvalue:
        fitvalue.append(h1/(cmax+0.001-i))
    return fitvalue
def sum(fitvalue):
    total = 0
    for i in range(len(fitvalue)):
        total += fitvalue[i]
    return total

# 自然选择（轮盘赌算法）
#输入种群中各个体适应值、个体浮点数编码
#自然选择后的个体浮点数编码
def selection(pop, fitvalue):
    newfitvalue = []
    totalfit = sum(fitvalue)
    for i in range(len(fitvalue)):
        newfitvalue.append(fitvalue[i] / totalfit)
    ms = [];
    poplen = len(pop)
    for i in range(poplen):
        ms.append(random.random())  # random float list ms
    ms.sort()
    fitin = 0
    newin = 0
    newpop = pop
    newfitvalue=cumsum(newfitvalue)
    while newin < poplen:
        newfitvalue[fitin]
        if (ms[newin] < newfitvalue[fitin]):
            newpop[newin] = pop[fitin]
            newin = newin + 1
        else:
            fitin = fitin + 1
    pop = newpop
    return pop
# 个体间交叉，实现基因交换
def crossover(pop, pc):
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
# 基因突变
def mutation(pop, pm):
    px = len(pop)
    py = len(pop[0])
    for i in range(px):
        if (random.random() < pm):
            mpoint = random.randint(0, py - 1)
            if (pop[i][mpoint] == '1'):
                if mpoint==0:
                    pop[i]='0'+pop[i]
                elif mpoint ==px-1:
                    pop[i]=pop[i]+'0'
                else:
                    pop[i]=pop[i][:mpoint] +'0'+pop[i][mpoint+1:]
            elif(pop[i][mpoint] == '0'):
                if mpoint==0:
                    pop[i]='1'+pop[i]
                elif mpoint ==px-1:
                    pop[i]=pop[i]+'1'
                else:
                    pop[i]=pop[i][:mpoint] +'1'+pop[i][mpoint+1:]
            else:
                pass
    return pop
if __name__ == "__main__":
    #参数初始化
    popsize = 50  # 种群的大小
    env_dir,baf_dir=parse_args('cfg.txt')#读配置文件获取获数据地址和写结果地址
    print('read file:',env_dir)
    data = pd.read_csv(env_dir,index_col=0,sep='\t')
    factor=list(data.columns[3:])
    factor_num=len(factor)#变量个数
    results = []
    bestindividual = []
    bestfit = 0
    pop=[]
    rlst=[_ for _ in range(len(data))]
    rand_index=random.sample(range(len(rlst)), popsize)
    for i in range(popsize):
        p = ''
        item=list(data[fc][rand_index[i]] for fc in factor)
        print(item)
        for k,j in enumerate(item):
            p += mybin(j)
        pop.append(p)
    print('pop',pop[-1],pop)
    for i in range(50):  # 繁殖epoch代
        pc = 0.6 / ((i + 5) // 5)  # 两个个体交叉的概率
        pm = 0.2 / ((i + 5) // 5)  # 基因突变的概率
        print('poplen', len(pop), len(pop[0]),sum([len(_) for _ in pop]),len(mybin(-11)))
        print('epoch=',str(i))
        #print('pop',len(pop))
        objvalue = calobjvalue(pop,data,factor)  # 计算目标函数值
        print('obj',objvalue)
        fitvalue = calfitvalue(objvalue)   # 计算个体的适应值
        print('fit',fitvalue)
        (bestindividual, bestfit)= best(pop, objvalue)  # 选出最好的个体和最好的函数值
        #print('best',bestindividual,bestfit)
            #print('exept',bestindividual)# 每次繁殖，将最好的结果记录下来
        pop=selection(pop, fitvalue)  # 自然选择，淘汰掉一部分适应性低的个体
        pop=crossover(pop, pc)  # 交叉繁殖
        pop=mutation(pop, pm)  # 基因突变
        result=split_dec(bestindividual)
        print('最高适应度为：',bestfit)
        print('选股条件：',result)
    with open(baf_dir,'a') as e:
        for idx,fc in enumerate(factor) :
            e.write(str(fc)+','+str(result[idx])+'\n')

