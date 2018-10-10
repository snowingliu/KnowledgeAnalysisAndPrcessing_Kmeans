"""
Created on  10 02:16:06 2018

@author: Xueying 51184507006
"""

import math
import time
import pandas as pd
from random import randint
# 数据清洗
def washData(inputData):
    msg=list()
    for c in inputData["content"][0:2000]:
        strList=c.split(" ")
        msg.append(strList)
    stopwordsFiles=open("stopwords.txt","r",encoding='UTF-8')
    stopwords=stopwordsFiles.read().splitlines()

    for strlst in msg:
        for i in range(len(strlst)-1,-1,-1):
            if ((len(strlst[i])<3) or (strlst[i] in stopwords)):
                strlst.pop(i)
    return msg
# 二维数组转化为一维数组
def mat2Vect(msg):
    z=list()
    for _List in msg:
        for wd in _List:
            z.append(wd)
    return z
# 转为词向量
def getEntityVector(mainVector,strList):
    entityVector={}
    for wd in mainVector:
        tmp=[0 for _ in range(len(mainVector))]
        tmp[mainVector.index(wd)]=1
        entityVector[wd]=tmp
    return entityVector

def getKeyPairVector(trainData,mainVector,strList,entityVector):
    kp={}
    for i in range(len(strList)):
        x,y=trainData["e1"][i],trainData["e2"][i]
        tmp=[0 for _ in range(len(mainVector))]
        for wd in strList[i]:
            tmp=addElements(tmp,entityVector[wd])
        kp[(x,y)]=tmp
    return kp

def addElements(vec1,vec2):
    tmp=list()
    if len(vec1)==len(vec2):
        for i in range(len(vec1)):
            tmp.append(vec1[i]+vec2[i])
        return tmp
    else:
        return None
def norm(vec):
    c=0
    for x in vec:
        c+=x*x
    return math.sqrt(c)

#计算余弦相似度
def cosine_similarity(vec1,vec2):
    numerator=0
    for i in range(min(len(vec1),len(vec2))):
        numerator+=vec1[i]*vec2[i]
    denominator=norm(vec1)*norm(vec2)
    return numerator/denominator

#检验
def getKeyPairType(trainData,strList):
    tpd=[]
    for i in range(len(strList)):
        x,y=trainData["e1"][i],trainData["e2"][i]
        tpd.append(trainData["type"][i])
    return tpd



def wpVector(x,y):
    return addElements(entityVector[x],entityVector[y])

if __name__=="__main__":
    
    startTime=time.clock()
    # 数据预处理
    trainData=pd.read_csv("finaltrain.csv")
    strList=washData(trainData)
    mainVector=mat2Vect(strList)
    entityVector=getEntityVector(mainVector,strList)
    kpVector=getKeyPairVector(trainData,mainVector,strList,entityVector)
    # 设置四个簇
    num_class=4
    cent=[None]*num_class
    centGroup=[cent,[None]*num_class]

    upperBound=-1
    for i in range(len(strList)):
        x,y=trainData["e1"][i],trainData["e2"][i]
        upperBound=max(max(kpVector[(x,y)]),upperBound)

    k=0
    RUNTIMES=0
    for i in range(num_class):
        centGroup[0][i]=[randint(0,upperBound) for _ in range(len(mainVector))]
        
    # 计算余弦相似度
    while RUNTIMES<20:
        contentClass=[[] for _ in range(num_class)]
        for i in range(len(strList)):
            x,y=trainData["e1"][i],trainData["e2"][i]
            maxValue=0
            tmpClassIndex=0
            for j in range(num_class):
                cos_Distance=cosine_similarity(centGroup[k][j],kpVector[(x,y)])
    
                if cos_Distance>maxValue:
                    maxValue=cos_Distance
                    tmpClassIndex=j
            contentClass[tmpClassIndex].append((x,y))
        
        print(RUNTIMES)
        for i in range(num_class):
            print(len(contentClass[i]),end="|")
        print()
        # 更新质心
        for i in range(num_class):
            centGroup[1-k][i]=[0 for _ in range(len(mainVector))]
            for wdpair in contentClass[i]:
                centGroup[1-k][i]=addElements(centGroup[1-k][i],kpVector[(wdpair[0],wdpair[1])])

            if(len(contentClass[i])>0):
                for x in range(len(centGroup[1-k][i])):
                    centGroup[1-k][i][x]=centGroup[1-k][i][x]/len(contentClass[i])
            else:
                for x in range(len(centGroup[1-k][i])):
                    centGroup[1-k][i][x]=centGroup[k][i][x]
        
        bias=0
        for i in range(num_class):
            bias+=cosine_similarity(centGroup[k][i],centGroup[1-k][i])
        print(bias)
        if(bias>num_class*0.99999):
            break
        k=1-k
        RUNTIMES+=1
    k=1-k
    cosdis=[]
    for i in range(num_class):
        cosdis.append([])
        for j in range(num_class):
            con_Distance=cosine_similarity(centGroup[k][j],centGroup[k][i])
            cosdis[i].append(cos_Distance)

    for i in cosdis:
        print(cosdis)
    
    #结果输出到.csv    
    target = pd.DataFrame(contentClass)
    target.to_csv("target.csv")
   # for ctx in contentClass:
   #     print(ctx,end='\n\n')
    #print("RUNTIME=%f sec."%(time.clock()-startTime))
