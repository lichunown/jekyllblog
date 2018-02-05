---
layout:     post
title:      "python networkx 笔记"
subtitle:   "至少比自己手写迪杰斯特拉强"
date:       2018-2-5
author:     "LCY"
header-img: "img/networkx.png"
tags:
    - python
    - 图论
    - networkx
---
## 基本操作

### 添加图
```python
import networkx as nx

g = nx.Graph()
g.add_edge(u,v,**kwargs)
```
![type of graph](/img/in-post/networkx/g_add.png)

networkx中的图，可以随意的添加附加信息， `g.node`&`g.edge`返回的是一个包含有给与信息的字典。

![type of graph](/img/in-post/networkx/typeofg.png)

### 度
```python
g.degree()
# or 
nx.degree(g)
```
返回字典，节点id ----> 节点的度

### 图论算法
常用图论算法，参考官方文档

[Networkx Algorithms](https://networkx.github.io/documentation/networkx-1.10/reference/algorithms.html)





## 特殊网络生成
- 规则图
  `random_graphs.random_regular_graph(d, n)`
- ER随机图
  `nx.random_graphs.erdos_renyi_graph(20, 0.2)`
- WS小世界网络
  `nx.random_graphs.watts_strogatz_graph(20, 4, 0.3)`
- BA无标度网络
  `random_graphs.barabasi_albert_graph(n, m)`

## 画图
- 无节点标号的简单图
```python
nx.draw(G, pos=None, ax=None, hold=None, **kwds)
```
- 有节点标号
```python
nx.draw_networkx(G, pos=None, with_labels=True, **kwds)
```
### 颜色&大小

```python
nx.draw(self.g,pos = self.pos, node_size = [x*30 for x in nx.degree(self.g).values()],
                        edge_color = 'gray', 
        				node_color = ['black' if i else 'r' for i in self.visited],
        				alpha = 0.8)
```

- edge_color：边的颜色，列表或数字

- node_color：节点颜色，列表或数字

- alpha：透明度，仅数字

- node_size：节点大小，列表或数字

  ![](/img/in-post/networkx/graphexample.png)

### pos(layout)

通过一定算法对图的位置进行计算，确定合理的布局。计算量比较大，多个图绘制时可保存此数据，加快绘图速度。

| layout                                   | 介绍                                       |
| ---------------------------------------- | ---------------------------------------- |
| [`circular_layout`](http://networkx.readthedocs.io/en/networkx-1.11/reference/generated/networkx.drawing.layout.circular_layout.html#networkx.drawing.layout.circular_layout)(G[, dim, scale, center]) | Position nodes on a circle.              |
| [`fruchterman_reingold_layout`](http://networkx.readthedocs.io/en/networkx-1.11/reference/generated/networkx.drawing.layout.fruchterman_reingold_layout.html#networkx.drawing.layout.fruchterman_reingold_layout)(G[, dim, k, ...]) | Position nodes using Fruchterman-Reingold force-directed algorithm. |
| [`random_layout`](http://networkx.readthedocs.io/en/networkx-1.11/reference/generated/networkx.drawing.layout.random_layout.html#networkx.drawing.layout.random_layout)(G[, dim, scale, center]) | Position nodes uniformly at random.      |
| [`shell_layout`](http://networkx.readthedocs.io/en/networkx-1.11/reference/generated/networkx.drawing.layout.shell_layout.html#networkx.drawing.layout.shell_layout)(G[, nlist, dim, scale, center]) | Position nodes in concentric circles.    |
| [`spring_layout`](http://networkx.readthedocs.io/en/networkx-1.11/reference/generated/networkx.drawing.layout.spring_layout.html#networkx.drawing.layout.spring_layout)(G[, dim, k, pos, fixed, ...]) | Position nodes using Fruchterman-Reingold force-directed algorithm. |
| [`spectral_layout`](http://networkx.readthedocs.io/en/networkx-1.11/reference/generated/networkx.drawing.layout.spectral_layout.html#networkx.drawing.layout.spectral_layout)(G[, dim, weight, scale, center]) | Position nodes using the eigenvectors of the graph Laplacian. |

### 批量输出保存

使用`plt.figure`等方法

```python
import ...
import matplotlib.pyplot as plt

g = createG(node_num) # networkx.Graph
pos = nx.spring_layout(g)
for i in range(nums):
    img = plt.figure(figsize=(15,15)) #单位英寸
    nx.draw(g,**kwargs)
    img.savefig('{}_{}.jpg'.format(name,i), dpi=100)
    g = createG(node_num)# next graph
    
```



## 附录

### 其他博文

[[python复杂网络库networkx：绘图draw](http://blog.csdn.net/pipisorry/article/details/54291831)

](http://blog.csdn.net/pipisorry/article/details/54291831)

### 官方文档
[NetworkX Overview](https://networkx.github.io/documentation/networkx-1.9/overview.html)
### 数据集
[Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/index.html)

- [Social networks](https://snap.stanford.edu/data/index.html#socnets) : online social networks, edges represent interactions between people
- [Networks with ground-truth communities](https://snap.stanford.edu/data/index.html#communities) : ground-truth network communities in social and information networks
- [Communication networks](https://snap.stanford.edu/data/index.html#email) : email communication networks with edges representing communication
- [Citation networks](https://snap.stanford.edu/data/index.html#citnets) : nodes represent papers, edges represent citations
- [Collaboration networks](https://snap.stanford.edu/data/index.html#canets) : nodes represent scientists, edges represent collaborations (co-authoring a paper)
- [Web graphs](https://snap.stanford.edu/data/index.html#web) : nodes represent webpages and edges are hyperlinks
- [Amazon networks](https://snap.stanford.edu/data/index.html#amazon) : nodes represent products and edges link commonly co-purchased products
- [Internet networks](https://snap.stanford.edu/data/index.html#p2p) : nodes represent computers and edges communication
- [Road networks](https://snap.stanford.edu/data/index.html#road) : nodes represent intersections and edges roads connecting the intersections
- [Autonomous systems](https://snap.stanford.edu/data/index.html#as) : graphs of the internet
- [Signed networks](https://snap.stanford.edu/data/index.html#signnets) : networks with positive and negative edges (friend/foe, trust/distrust)
- [Location-based online social networks](https://snap.stanford.edu/data/index.html#locnet) : Social networks with geographic check-ins
- [Wikipedia networks, articles, and metadata](https://snap.stanford.edu/data/index.html#wikipedia) : Talk, editing, voting, and article data from Wikipedia
- [Temporal networks](https://snap.stanford.edu/data/index.html#temporal) : networks where edges have timestamps
- [Twitter and Memetracker](https://snap.stanford.edu/data/index.html#twitter) : Memetracker phrases, links and 467 million Tweets
- [Online communities](https://snap.stanford.edu/data/index.html#onlinecoms) : Data from online communities such as Reddit and Flickr
- [Online reviews](https://snap.stanford.edu/data/index.html#reviews) : Data from online review systems such as BeerAdvocate and Amazon