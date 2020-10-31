import math
import operator
import pickle
traindata = []
testdata = []
removed = []

def read(path,buf = []):
    with open(path,"r") as fp:
        for line in fp.readlines()[1:-1]:
            line = line.split("\t")
            line[-1] = line[-1][0:-1]
            line = [float(x) for x in line]
            line[-1] = int(line[-1])
            buf.append(line)

#求某一特征集合的信息熵
def getEntropy(Set):
    CategoryCnt = {}
    for temp in Set:
        tempCategory = temp[-1]
        if tempCategory not in CategoryCnt.keys():
            CategoryCnt[tempCategory] = 0
        CategoryCnt[tempCategory] +=1

    entropy = 0.0
    num = len(Set)
    for key in CategoryCnt:
        p = float(CategoryCnt[key])/num
        entropy -= p * math.log2(p)

    return entropy

#将xi大于或小于value的特征划分进不同的集合内并返回
def splitSet(Set, i, value):
    setless = []
    setgreater = []
    for feature in Set:
        tempfeature = feature[:i]
        tempfeature.extend(feature[i+1:])
        if feature[i]<value:
            setless.append(tempfeature)
        else:
            setgreater.append(tempfeature)

    return [setless,setgreater]


#第i个特征的最小条件熵 返回minEntropy和xi的划分取值 求最小条件熵即最大信息增益
def getConditionalEntropy(Set, i):
    tempSet = Set
    min =  1000000.0
    value = 0
    for feature in tempSet:
        ent = 0.0
        setList = splitSet(Set,i,feature[i])
        for newset in setList:
            p = len(newset)/len(Set)
            ent += p * getEntropy(newset)
        if min>ent:
            min = ent
            value = feature[i]

    return [min,value]


#对某一节点测试所有特征属性xi 找到信息增益最大的
def attributeTest(Set):
    attributeCount = len(Set[0])-1
    entropy = getEntropy(Set)
    maxGain = 0.0
    attributeNum = -1
    value = -1
    for i in range(attributeCount):
        tempList = getConditionalEntropy(Set,i)
        tempEntropy = tempList[0]
        tempValue = tempList[1]
        tempGain = entropy-tempEntropy
        if tempGain>maxGain:
            maxGain = tempGain
            value = tempValue
            attributeNum = i
    return [attributeNum,value]


#如果特征已经用完，选择特征集合中多数的结果作为结果
def judgebyMajority(resultList):
    tempmap = {}
    for i in resultList:
        if i not in tempmap.keys():
            tempmap[i] = 0
        tempmap[i] += 1
    return sorted(tempmap.items(),key=operator.itemgetter(1),reverse=True)[0][0]


def buildTree(Set):
    resultList = [i[-1] for i in Set]
    if resultList.count(resultList[0]) == len(resultList):
        return resultList[0]
    if len(Set[0]) == 1:
        return judgebyMajority(resultList)

    attributeList = attributeTest(Set)
    attributeNum = attributeList[0]
    removed.sort()
    for i in removed:
        if i<= attributeNum:
            attributeNum += 1
    removed.append(attributeNum)

    attributeValue = attributeList[1]
    Tree = {(attributeNum,attributeValue):{}}
    for i in range(2):
        setlist = splitSet(Set,attributeList[0],attributeValue)
        Tree[(attributeNum,attributeValue)][i] = buildTree(setlist[i])
    return Tree


def test(Tree,testdata):
    attrList = list(Tree.keys())[0]
    nextDic = Tree[attrList]
    attrNum = attrList[0]
    attrValue = attrList[1]

    flag = 1 if testdata[attrNum]>=attrValue else 0
    if type(nextDic[flag]).__name__ == 'dict':
        classification = test(nextDic[flag],testdata)
    else:
        classification = nextDic[flag]
    return classification


def serialize(path):
    fp = open(path,"wb")
    pickle.dump(Tree,fp)
    fp.close()


if __name__ == "__main__":
    read("traindata.txt",traindata)
    read("testdata.txt",testdata)

    Tree = buildTree(traindata)
    correctNum = 0

    for testFeature in testdata:
        classification = test(Tree,testFeature)
        if classification == testFeature[-1]:
            correctNum += 1
        print(classification,end=' ')
    print()

    print("test result: accuracy is {}%".format(correctNum/len(testdata)*100))
    serialize("./obj/tree.txt")
