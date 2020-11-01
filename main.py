import math
import operator
import pickle

# 训练数据
traindata = []
# 测试数据
testdata = []
# 已经移出的特征的原索引号
removed = []


# 读文件
def read(path, buf):
    with open(path, "r") as fp:
        for line in fp.readlines()[1:-1]:  # 不读取第一行和最后一行 因为为[和]
            line = line.split("\t")  # 以Tab键分裂字符串
            line[-1] = line[-1][0:-1]  # 最后一项为'1\n' 需要去掉\n
            line = [float(x) for x in line]  # 类型转换
            line[-1] = int(line[-1])
            buf.append(line)


# 求某一特征集合的信息熵
def getEntropy(Set):
    # 对各个种类的进行计数，以便计算信息熵
    CategoryCnt = {}
    for temp in Set:
        tempCategory = temp[-1]
        if tempCategory not in CategoryCnt.keys():
            CategoryCnt[tempCategory] = 0
        CategoryCnt[tempCategory] += 1

    entropy = 0.0
    num = len(Set)
    for key in CategoryCnt:
        p = float(CategoryCnt[key]) / num
        entropy -= p * math.log2(p)

    return entropy


# 将xi大于或小于value的特征划分进不同的集合内并返回
# i表示列索引值， value表示分裂的值
# 该函数用于分裂节点求以及条件熵
def splitSet(Set, i, value):
    setless = []
    setgreater = []
    for feature in Set:
        tempfeature = feature[:i]
        tempfeature.extend(feature[i + 1:])
        # 分裂规则：小于或大于等于
        if feature[i] < value:
            setless.append(tempfeature)
        else:
            setgreater.append(tempfeature)

    return [setless, setgreater]


# 第i个特征的最小条件熵 求最小条件熵即最大信息增益
# 返回minEntropy和xi的划分取值
# 由于该树采用二叉树结构，所以必须对同一属性的所有值全部求条件熵并找到条件熵最小的属性值
def getConditionalEntropy(Set, i):
    tempSet = Set
    min = 1000000.0
    value = 0
    for feature in tempSet:
        ent = 0.0
        # 分裂并求条件熵
        setList = splitSet(Set, i, feature[i])
        for newset in setList:
            p = len(newset) / len(Set)
            ent += p * getEntropy(newset)
        if min > ent:
            min = ent
            value = feature[i]
    # 返回的是最小条件熵以及得到该条件熵的属性值
    # 即分裂处的取值
    return [min, value]


# 对某一节点测试所有特征属性xi 找到信息增益最大的
def attributeTest(Set):
    # 剩余的属性个数
    attributeCount = len(Set[0]) - 1
    # 求现在状态的信息熵
    entropy = getEntropy(Set)
    # 定义信息增益
    maxGain = 0.0
    # 定义属性索引和值
    attributeNum = -1
    value = -1
    for i in range(attributeCount):
        tempList = getConditionalEntropy(Set, i)
        tempEntropy = tempList[0]
        tempValue = tempList[1]
        tempGain = entropy - tempEntropy
        # 找到信息增益最大的属性
        if tempGain > maxGain:
            maxGain = tempGain
            value = tempValue
            attributeNum = i
    # 返回属性的索引和值
    return [attributeNum, value]


# 如果特征已经用完，选择特征集合中多数的结果作为结果
def judgebyMajority(resultList):
    tempmap = {}
    for i in resultList:
        if i not in tempmap.keys():
            tempmap[i] = 0
        tempmap[i] += 1
    # 返回结果需要是一个值，该值作为字典的value
    # 这是由树的数据结构决定的
    return sorted(tempmap.items(), key=operator.itemgetter(1), reverse=True)[0][0]


# 构造决策树的过程
def buildTree(Set):
    # 递归调用的特征集合中的特征列表
    resultList = [i[-1] for i in Set]
    # 如果特征列表只有一种值，即已经完成划分，那么直接返回该值
    if resultList.count(resultList[0]) == len(resultList):
        return resultList[0]
    # 如果属性已经用完了，那么需要根据多数结果判断
    if len(Set[0]) == 1:
        return judgebyMajority(resultList)

    attributeList = attributeTest(Set)
    attributeNum = attributeList[0]
    attributeValue = attributeList[1]

    # 由于删除属性后，属性的索引值会改变，所以需要定义removed并修改attributeNum
    removed.sort()
    for i in removed:
        if i <= attributeNum:
            attributeNum += 1
    removed.append(attributeNum)

    # 树结构的定义，Key为包含属性索引和值的Tuple value为递归后返回的字典(未到子节点)或结果值(已经到子节点)
    Tree = {(attributeNum, attributeValue): {}}
    # 分别对大于小于两种情况递归建树
    for i in range(2):
        setlist = splitSet(Set, attributeList[0], attributeValue)
        Tree[(attributeNum, attributeValue)][i] = buildTree(setlist[i])
    # 递归返回新树
    return Tree


# 测试过程是进行一个递归进行树搜索的过程
def test(Tree, testdata):
    attrList = list(Tree.keys())[0]
    # 下一个递归过程需要用到的树结构(即字典中现在key的value)
    nextDic = Tree[attrList]
    attrNum = attrList[0]
    attrValue = attrList[1]

    # 大于等于决策树节点值向右找，小于向左找
    flag = 1 if testdata[attrNum] >= attrValue else 0
    # 如果nextDic[flag]仍是字典，表明仍未递归到子节点
    if isinstance(nextDic[flag],dict):
        classification = test(nextDic[flag], testdata)
    # 如果已经是子节点，则找到了分类
    else:
        classification = nextDic[flag]
    return classification


# 使用pickle对该树进行序列化，并使用序列化文件和graphviz画图
def serialize(path):
    fp = open(path, "wb")
    pickle.dump(Tree, fp)
    fp.close()


if __name__ == "__main__":
    read("traindata.txt", traindata)
    read("testdata.txt", testdata)

    correctNum = 0
    wrong_result = []

    Tree = buildTree(traindata)

    for testFeature in testdata:
        classification = test(Tree, testFeature)
        # 分类正确
        if classification == testFeature[-1]:
            correctNum += 1
        # 记录错误数据
        else:
            tmp = []
            tmp.append(testFeature[:-1])
            tmp.append(classification)
            tmp.append(testFeature[-1])
            wrong_result.append(tmp)
        print(classification, end=' ')
    print()
    for tmp in wrong_result:
        print("WRONG result: feature is {} | expected result is kind {} | real result is kind {}".format(tmp[0], tmp[2], tmp[1]))
    print("test result: accuracy is {}%".format(correctNum / len(testdata) * 100))
    serialize("./serialize/tree.txt")
