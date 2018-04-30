---
layout:     post
title:      "图编码算法汇总"
subtitle:   "没错, 全是外链"
date:       2018-4-30
author:     "LCY"
header-img: "img/default.jpg"
tags:
    - DL
    - 图论
    - python
---

## 写在前面

近期看了好多图编码的算法,做个小小的总结.
(知乎是个好东西)

## 图卷积(GCN)
GCN是一种类比图片卷积操作,而产生的图编码算法.
关于GCN的推导,知乎上有个回答已经很详细了[如何理解 Graph Convolutional Network(GCN)](https://www.zhihu.com/question/54504471)

首先是拉普拉斯矩阵, 最简单的定义为:
$$
L=D-A
$$
其中, $D$为图的对角阵, $A$为图的邻接矩阵.
在机器学习中,为了简化计算的复杂度(具体的化简方法参考知乎的链接), 将卷积简化为了:
$$
P=\sum_{k=0}^{\infty}\alpha(1-\alpha)^k(D^{-1}W)^k
$$
$k$就是卷积的步数(diffusion step). 

这个公式可以怎么理解呢? 首先,$D^{-1}W$可以堪称由节点i, 传播到其他所有节点的一个概率. 而($D^{-1}W)^2$,其实也就是$\sum^n_{k=1}w_{ik}w_{kj}$, 可以看作, 由节点i,通过k, 到达j的一个传播概率.

#### 神经网络里的图卷积层

神经网络里的图卷积操作可定义为:
$$
X_{:,p*G\ }f_\theta=\sum_{k=0}^{K-1}(\theta_{k,1}(D^{-1}W)^k)X_{:,p}
$$

由于在实际操作中,$K$不能无限大, 因此必须把$K$限制在一定范围内, 一般而言, $K=3,4$, 就够了.

而神经网络中的图卷积层就可以定义为:

$$
H_{:,q}=a(\sum_{p=1}^PX_{:,p*G\ }f_\theta)\ \ \ \ for\ q\in\{1,...,Q\}
$$

对于有向图来说, 可以将图的出度和入度区分开, 这样, 图卷积就是:
$$
X_{:,p*G\ }f_\theta=\sum_{k=0}^{K-1}(\theta_{k,1}(D_O^{-1}W)^k+\theta_{k,2}(D_I^{-1}W^T)^k) X_{:,p}
$$
$D_O$, $D_I$分别表示图的出度拉普拉斯矩阵和入度的拉普拉斯矩阵. 


## DeepWalk这一类

DeepWalk这的类方法, 借鉴了自然语言处理中的word2vec方法. 大题思路就是通过一定的规则, 随机游走整个图, 走过的节点构成一个序列, 这样的序列就作为"一句话", 很多句话传入word2vec模型中,得到训练生成的特征向量.

### 关于word2vec
详细介绍参考:[一篇通俗易懂的word2vec](https://zhuanlan.zhihu.com/p/35500923)
word2vec大致有两种算法, 一为CBOW,  一为SkipGram. 在图编码中, 貌似skip gram算法用的最多.
![skip gram算法图示](/img/in-post/graph-encoding/skip.jpg)

总之, 训练的结果, 两个相似词之间的cos值最小. 并且会产生很多有意思的结果, 比如:

$vec(国王)+vec(女人)-vec(男人)$ is close to $ vec(女王)$

下图是训练后的词向量, 在二维情况下的映射关系
![word2vec 词向量分析](/img/in-post/graph-encoding/learn.png)

### DeepWalk
详细介绍的连接:[论文阅读：DeepWalk](http://www.cnblogs.com/lavi/p/4323691.html)

思想很简单, 就是上面说的, 采样就是随机游走. 

### node2vec

同样是外链:[node2vec: Scalable Feature Learning for Networks 阅读笔记](https://zhuanlan.zhihu.com/p/30599602)
在DeepWalk的基础上, 通过引入参数来控制游走路径, 就得到了我们的新算法node2vec. 

![node2vec 实例](/img/in-post/graph-encoding/node2vec.jpg)
对于随机游走，一般的做法是直接用边的权重来转化成概率，本文作者在权重之前乘上一个系数$\alpha$.
如图, 我们引入两个参数$p$, $q$. 我们假定节点在此刻由t游走到了v. 
那我们定义$d$, 是v的邻居节点到上一节点t之间的距离.
那么新的路径概率$\alpha$:
$$
\alpha=
\left \{\begin{array}
          \frac^1_p,\  d=0 \\
          1, \ d=1  \\
          \frac1q, \ d=2 
       \end{array}
\right .
$$
通过调节参数 p 和 q ，可以控制随机游走的方向，两个极端的方向就是BFS和DFS，也就是一直在节点 t 旁边绕或者一直往外面走，这样就能做到一个比较灵活的调节。

### struc2vec
继续外链:[STRUC2VEC（图结构→向量）论文方法解读](http://jackieanxis.coding.me/2018/01/17/STRUC2VEC/)

这个方法就很神奇了啊, 它在原图的基础上, 扩展了一个叫上下文图的东西(context graph).
原本一个平面的图结构,变成了一个立体的图结构.
![struc2vec](/img/in-post/graph-encoding/struc2vec.png)
其中, 第k层图表达的是节点之间k级邻域的相似度度量.

### graph2vec
在word2vec产生后, 很快就有了doc2vec. word2vec是词向量编码, 而doc2vec则是对每一句话编码.
[Doc2Vec模型介绍及使用](https://blog.csdn.net/Walker_Hao/article/details/78995591)

同样的,类比于doc2vec, 我们有了对整个图进行编码的graph2vec
这个我并没有细看, 直接贴论文吧.
[graph2vec: Learning Distributed Representations of Graphs](https://arxiv.org/abs/1707.05005)

## PATCHY-SAN
这是一个很偏门的方法, 论文题目叫[Learning Convolutional Neural Networks for Graphs](http://proceedings.mlr.press/v48/niepert16.pdf), 下意识的我还以为是GCN呢. 于是顺带着瞄了几眼. 这个方法很别扭, 似乎也没有开源实现, 感觉有点.........不太靠谱??
总之,还是先贴外链[论文笔记：Learning Convolutional Neural Networks for Graphs](https://zhuanlan.zhihu.com/p/27587371)
![PATCHY-SAN](/img/in-post/graph-encoding/PATCHY1.jpg)
说白了, 就是选出固定数量的点, 然后再对每个点选取它们的邻居, 按照一定规律, 对每一堆点分别标号. 
![PATCHY-SAN](/img/in-post/graph-encoding/PATCHY2.jpg)
再由这些标好号的点, 组成一组张量, 对这组张量进行卷积操作.

感觉这个方法怪怪的.

## GNN类

这又是好多种方法, 不过我都没怎么细看
再次直接贴连接:[《Gated Graph Sequence Neural Networks》阅读笔记](https://zhuanlan.zhihu.com/p/28170197)
### GNN
![PATCHY-SAN](/img/in-post/graph-encoding/GNN.jpg)
首先输入一个图, 对每个节点进行一系列的encoding变换, 最后输出一个特定维度的向量. 怎么看都感觉和GCN好像啊.

### GG-NN
> 相比于GNN，GG-NN的特点在于使用了GRU单元，而且，节点表示的更新次数被固定成了 T 。也就是说，GG-NN并不能保证图的最终状态会到达不动点。由于更新次数 T 变成了固定值，因此GG-NN可以直接使用BPTT算法来进行梯度的计算。相比于Almeida-Pineda算法，BPTT算法需要更多的内存，但是，它并不需要对参数进行约束(以保证收敛)。

### GGS-NN
> GG-NN一般只能处理单个输出。若要处理输出序列 $o^{(1)},...,o^{(K)} $，可以使用GGS-NN（Gated Graph Sequence Neural Networks）。

## Graph Attention Networks(GAT)
这个是今年的新算法, 紧跟一波attention network潮流. 
首先看一眼attention network.
![PATCHY-SAN](/img/in-post/graph-encoding/attention.jpg)

再看一眼别人的笔记:[《Graph Attention Networks》阅读笔记](https://zhuanlan.zhihu.com/p/34232818)
![PATCHY-SAN](/img/in-post/graph-encoding/gat.jpg)

总之, 和GCN也是非常相似. 另外, 这种算法直接考虑图节点的feature, 图直接连接的权重, 似乎直接忽略了?? 训练的时候直接就记忆进W了??
真的不是很懂, 还要继续研读. 
论文链接: [Graph Attention Networks](https://arxiv.org/abs/1710.10903)

##End
最后贴一波论文
[Diffusion-Convolutional Neural Networks](https://arxiv.org/abs/1511.02136)
[DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)
[node2vec: Scalable Feature Learning for Networks](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf)
[struc2vec: Learning Node Representations from Structural Identity](https://arxiv.org/abs/1704.03165)
[graph2vec: Learning Distributed Representations of Graphs](https://arxiv.org/abs/1707.05005)
(PATCHY-SAN)[Learning Convolutional Neural Networks for Graphs](http://proceedings.mlr.press/v48/niepert16.pdf)
[Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493)
[Structured Attention Networks](https://arxiv.org/abs/1702.00887)
[Graph Attention Networks](https://arxiv.org/abs/1710.10903)