---
layout:     post
title:      "神经网络的入门教程"
subtitle:   "上周技术沙龙的某些总结"
date:       2017-05-24
author:     "LCY"
header-img: "img/default.jpg"
tags:
    - ML
---



## 线性回归
### 引子
![](/img/in-post/nurealNetwork-salon/LinearRegressionImg.png)
忽略那条线，现在，我们有这样的一些数据。可以清楚的看到，这些数据点，似乎呈现一个线性分布。而高中的时候（数学统计那块的），我们学过，有一个叫最小二乘法的东西，带一下公式，就可以把这条拟合的曲线画出来。
然而，我们不用那个公式，（一步出结果多麻烦啊，还要公式推导证明。。。）有没有一个简单的通法呢？答案当然是有的（虽然计算量有点多，不过，我们有计算机啊）。

什么情况算拟合程度较好呢？高中的数学课本上给出了一种方法：**求方差**。

让我们假设这条直线是$$y=ax+b$$，假设数据对应的x为$$x_i$$，数据对应的y是$$y_i$$。数据量一共有m个，那我们有方差：
$$
J=\frac{1}{m} \sum_{i=1}^m{y(x_i)-y_i}
$$



我们怎么找出这个最好的a和b，让方差最小呢？很简单，求导。

。。。。。。

### 正文部分来了

如果我们用i啊j啊什么的写循环，那可真是太麻烦了。所以，在真正的代码中，我们都是用的一些矩阵变换。用矩阵的好处是，数学家还有一些计算机大牛们已经帮我们优化好了矩阵计算的算法，用矩阵计算的速度会大幅下降。

前面提到了直线是`y=ax+b`，然而在现实中，输入的变量可能是多个`x_1,x_2,x_i,...`。不过，如果我们把它写成矩阵的形式，就简单多了。
$$
Y=X\theta
$$
其中，$$\theta$$是一个列向量，代表上文所说的a，b。X的每一行是一个数据，有几个输入$$x_i$$，就有几列。（ax+b，这个b也是算一个$$\theta$$，对应的x取常数1即可。）

**在这里需要提一句，网上的教程常写作$$Y={\theta}^TX$$。然而，在实际应用中，$$\theta$$常取一个列向量，X的每一行，代表一组数据。在这种情况下，写成广为流传的形式。。绝对会出问题。**

为了求出这个最佳的$\theta$，有一种梯度下降方法，即不断的迭代$$\theta := \theta-d\theta$$。而这一方法，写成矩阵表达的话将是
$$
\theta=\theta-(1/m)X^T(X*\theta-Y)
$$

（至于怎么推导出来的……看其他的blog吧，码字好累……）















【未待完续】下面的我还会详写。。。





## logistic

然后，切入了本讲的主要内容，logistic回归。logistic回归从思想上来说，和线性回归是一样的。不同的是，它在进行$\theta X$运算之后，使用了一个sigmod函数，将输出y限制在了0到1之间。通过0和1，我们可以用这种回归去训练一个简单的二分类问题。

sigmod为：
$$
Y=\frac{1}{1+e^{-X \theta}}
$$
logistic回归的训练和线性回归相似，然而，在训练时，为了确保函数不是非凸函数，我们常常在最后加上log项。



```python
#encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt
#数据随便取的
data = [
        [1,2,1],
        [2,3,1],
        [0.5,2,1],
        [3,5,1],
        [3,4,1],
        [2,4,1],
        
        [2,1,0],
        [3,2,0],
        [2,0.5,0],
        [5,3,0],
        [4,3,0],
        [4,2,0],
    ]
data = np.array(data)
#初始化一个theta，值可以随便取
theta = np.array([
            [1],
            [5],
            [1],
        ],'float'
    )


x = np.concatenate((np.ones([data.shape[0],1]),data[:,[0,1]]/5),1)#ax+b啊，添加偏置项b（就是加上一列常数1）

y = np.array([data[:,2]]).T#输出的y是个列向量

# 画图
plt.plot(data[np.where(data[:,2]==1),0],data[np.where(data[:,2]==1),1],'r+')
plt.plot(data[np.where(data[:,2]==0),0],data[np.where(data[:,2]==0),1],'b*')

# sigmod函数
def logistic(x):
    return 1/(1+np.exp(-x))
# 输出的函数h（x）
def hFunction(theta,x):
    return logistic(np.dot(x,theta))
# 定义代价函数以及代价函数的导数
def costFunction(theta,x,m=1):
    # 不用这种方法，计算太麻烦
    #cost[np.where(data[:,2]==1),0] = -np.log(h[np.where(data[:,2]==1),0])
    #cost[np.where(data[:,2]==0),0] = -np.log(1-h[np.where(data[:,2]==0),0])
    cost = (1/m)*np.sum(-y*np.log(hFunction(theta,x))-(1-y)*np.log(1-hFunction(theta,x)))
    dJ = np.array([(1/m)*np.sum(np.dot(x.T,hFunction(theta,x)-y),1)]).T
    return cost,dJ
# 训练
def train(x=x,y=y,theta=theta,alpha=0.01):
    while True:
        cost,dJ = costFunction(theta,x)
        theta -= dJ*alpha
        if np.sum(dJ)<0.001:#代价函数的导数小于0.001退出循环
            break
    return theta,cost

# 画图
def plotLine(theta=theta,data=data):
    fx = lambda x:-theta[1]/theta[2]*x-theta[0]/theta[2]
    x = np.arange(0,5,0.1)
    y = fx(x)
    plt.hold(True)
    plt.plot(data[np.where(data[:,2]==1),0],data[np.where(data[:,2]==1),1],'r+')
    plt.plot(data[np.where(data[:,2]==0),0],data[np.where(data[:,2]==0),1],'b*')
    plt.plot(x,y)

#最后执行 
train()
plotLine()
```





紧接着，又讲解了和logistic回归有密切联系的softmax回归。logistic是为了解决二分类问题，而softmax可以解决多分类问题。它的公式是
$$
Y=\frac{1}{\sum{e^{X\theta_i}}}*[e^{X\theta_1} \ e^{X\theta_2}\ ...\ e^{X\theta_i}]
$$
这是一个多输出的函数，一般，有几个分类，我们设置几个输出。可以看出，对应这每一个输入，它的每一个输出都是除以输出之和的一个比率。在这里，我们认为，这个输出代表它对应分类的概率。

而对softmax优化，除了求导进行梯度下降以外，还有一种优化方式。即使用交叉熵为代价函数，进行一个优化。交叉熵的概念是基于香农提出的信息论，可以近似的把交叉熵理解为两个函数输出分布的差异。交叉熵越小，两个函数的异同就越小。因此，类似于线性回归，我们也可以对交叉熵求导进行梯度下降求最优化。而在求导的时候我们发现，对交叉熵的求导在逼近最优点的时候不存在梯度变化缓慢的现象。可以有效的减少迭代次数。

交叉熵：
$$
C=-\frac{1}{n}\sum_n{ylna+(1-y)ln(1-a)}
$$
最后，讲解了一下圣经网络的前向传播。有人说，可以把神经网络理解成一个大脑，每一个神经网络的节点对应着一个神经元。然而这样的描述太过抽象。如果简单的说起来，大家可以把神经网络理解为多层的logistic回归，每一层有多个logistic回归，上一层的回归结果作为下一层的输入继续进行logistic回归，在有限的层数内，最终实现一个输出。（仅仅是以logistic为激励函数的神经网络）

假定输入X是一个矩阵，那么第一层的输出$$A_1=g(X*W_1)$$。在这里，g（x）表示的是前面提到的logistic回归用到的sigmod函数。而下一层的运算可以表示为$$A_2=g(A_1*W_2)$$。就这样一层层的前向传播过去。如果合起来写的话，将是
$$
Y=g(g(g(X*W_1)*W_2)...)
$$
至于$$W_i$$的维度，可以通过简单的矩阵乘法法则推导出来。



最后，给大家看了一下神经网络的具体应用。即MNIST的手写识别过程。在那个简单的手写识别中，我使用了一个隐藏层的神经网络。隐藏层采用logistic激励，最后的输出采用softmax进行多分类。而代价函数使用的是交叉熵函数。通过google开源的tensorflow库，简单快捷的进行训练。