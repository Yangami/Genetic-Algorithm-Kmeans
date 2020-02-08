import numpy as np
import pandas as pd
import configparser
def parse_args(filename):
    cf = configparser.ConfigParser()
    cf.read(filename)
    test = cf.items("test")
    dic = {}
    for key, val in test:
        dic[key] = val
    testdir = dic['test_dir']
    train = cf.items("train")
    dic1 = {}
    for key, val in train:
        dic1[key] = val
    bafdir = dic1['baf_dir']
    return testdir,bafdir
test_dir,bafdir=parse_args('cfg.txt')
df=pd.read_csv(test_dir,index_col=0,sep='\t')
baf=open(bafdir).read().split('\n')
if len(baf[-1])==0:
    baf=baf[:-1]
bafdic={}
for i in baf:
    t=i.split(',')
    print(t)
    bafdic[t[0]]=float(t[1])

baf=[-21533395427.313644, -21675430890.61889, -4.374825448864781, -3259.521318071086, 254.3186405010215, -41.3919664145219, -0.30902633307824645, 5.581310464149584, 77.82204172375589]
pool=[]
for k in df.index:
    k=df.loc[k]
    flag=1
    #['total_assets', 'total_liability', 'roe', 'pcf_ratio', 'pe_ratio_lyr', 'net_profit_to_total_revenue', 'eps', 'pb_ratio', 'market_cap']
    for j in bafdic.keys():
        if float(k[j])<bafdic[j]:
            flag=0
    if flag==1:
        pool.append(k.name)
print(pool)