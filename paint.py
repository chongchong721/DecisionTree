from graphviz import Digraph
import pickle


# 求最大叶子节点
def getMaxLeafs(myTree):
    numLeaf = len(myTree.keys())
    for key, value in myTree.items():
        if isinstance(value, dict):
            sum_numLeaf = getMaxLeafs(value)
            if sum_numLeaf > numLeaf:
                numLeaf = sum_numLeaf
    return numLeaf


def plot_model(tree, name):
    g = Digraph("./paint/G", filename=name, format='png', strict=False)
    first_label = list(tree.keys())[0]
    # 先单独对根节点作图
    g.node("0", str(first_label))
    # 递归对后续节点作图
    _sub_plot(g, tree, "0",float(first_label[1]))
    leafs = str(getMaxLeafs(tree) // 10)
    g.attr(rankdir='LR', ranksep=leafs)
    g.view()


root = "0"


def _sub_plot(g, tree, inc, value):
    global root

    first_label = list(tree.keys())[0]
    ts = tree[first_label]
    for i in ts.keys():
        # 如果不是叶子节点
        if isinstance(tree[first_label][i], dict):
            root = str(int(root) + 1)
            # 画节点
            g.node(root, str(list(tree[first_label][i].keys())[0]))
            # 根据左右画边
            if i==0:
                mystr = ">="+str(value)
            else:
                mystr = "<"+str(value)
            g.edge(inc, root, str(mystr))
            _sub_plot(g, tree[first_label][i], root,list(tree[first_label][i].keys())[0][1])
        else:
            root = str(int(root) + 1)
            # 画节点
            g.node(root, str(tree[first_label][i]))
            # 根据左右画边
            if i==0:
                mystr = ">="+str(value)
            else:
                mystr = "<"+str(value)
            g.edge(inc, root, str(mystr))


def grabTree(fileName):
    fr=open(fileName, "rb")
    return pickle.load(fr)


if __name__ == "__main__":
    Tree = grabTree("./serialize/tree.txt")
    plot_model(Tree, "./paint/tree")