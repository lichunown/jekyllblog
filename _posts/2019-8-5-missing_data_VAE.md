---
layout:     post
title:      "《Handling Incomplete Heterogeneous Data using VAEs》论文笔记"
subtitle:   "只是整理&笔记"
date:       2019-8-5
author:     "LCY"
header-img: "img/default.jpg"
tags:
    - DL
    - VAE
    - 算法
---


论文连接：<https://arxiv.org/abs/1807.03653>

源码连接：<https://github.com/probabilistic-learning/HI-VAE>

## 数据分类

定义数据集中的第$n$条数据是$D$维的向量$x=[x_{n1},...,x_{nD}]$,参数$d$对应数据中的每一维。数据可以异构数据一般有如下情况。

- 数值变量
  - 实数数据，$$x_{nd}\in \mathbb{R}$$
  - 正实数数据，$$x_{nd}\in \mathbb{R}^+$$
  - 离散计数数据，$$x_{nd}\in \{0,1,...,\infty\}$$
- 名称变量
  - 分类数据，有限无序集合中的值，$$x_{nd}\in \{'blue','red','black'\}$$
  - 序数数据，有限有序集合的值，$$x_{nd}\in \{'never','sometime','often'\}$$

我们认为数据中的随机条目值不完整，每个对象$$x_n$$可能包含缺失数据和未缺失（可观测observed）数据的任意组合。令$$\mathcal{O}$$为可观测数据的索引集，$$\mathcal{M}$$为缺失数据的索引集。即$$\mathcal{O} \cap \mathcal{M} =\emptyset$$。令$$x_n^o$$是一条数据中可观测的切片，$$x_n^m$$是一条数据中缺失数据的切片。

与其他深度学习领域不同，我们的数据不是高度结构化数据，每个观察对象可能是数值变量与名称变量的混合。由于数据维度小（与图像相比），我们需要仔细设计生成模型以避免过拟合。

## 再探VAE

### 处理不完全数据

在标准VAE中，缺失数据会影响encoder和decoder模型。ELBO是在完整数据集上定义的，由于丢失的数据与其余数据不是直接分离的，特别是缺失数据在数据集中随机出现。我们对decoder有以下公式。

![1564985507829](/img/in-post/2019-8-5-missing_data_VAE/1564985507829.png)

![1564985870837](/img/in-post/2019-8-5-missing_data_VAE/1564985870837.png)



这里，$$z_n\in \mathbb{R}^K$$是encoder输出关于$$x_n$$的k维encode值，并且满足$$p(z_n)=\mathcal{N}(0,I_K)$$。这种因式分解允许容易地从变量ELBO中边缘化每个对象的缺失属性。用参数$\gamma_{nd}=h_d(z_n)$来参数化概率分布$$p(x_{nd}|z_n)$$。这里，$$h_d(z_n)$$是一个DNN网络，将encode的值转换为似然参数$$\gamma_{nd}$$。

![1564986923748](/img/in-post/2019-8-5-missing_data_VAE/1564986923748.png)

考虑到缺失数据，encode的数据$$z_n$$实际只依赖于观测数据$$x_n^o$$。即：

![1564987392701](/img/in-post/2019-8-5-missing_data_VAE/1564987392701.png)

![1564988962066](/img/in-post/2019-8-5-missing_data_VAE/1564988962066.png)

我们需要一个足够灵活的识别模型处理缺失数据和非缺失数据的任意组合，令$$\tilde{x}_n$$代表可观测的$d$维数据，其中未观测到的数据用0替换。有

![1564989574363](/img/in-post/2019-8-5-missing_data_VAE/1564989574363.png)

这里$$\mu_q(\tilde{x}_n)$$和$$\sum_q(\tilde{x}_n)$$是输出为均值和方差矩阵的DNN网络。

有一个替代方案，设计一个识别模型（factorized recognition model)：

![1564990614415](/img/in-post/2019-8-5-missing_data_VAE/1564990614415.png)

满足

![1564990631694](/img/in-post/2019-8-5-missing_data_VAE/1564990631694.png)

因此，公式

![1564990713245](/img/in-post/2019-8-5-missing_data_VAE/1564990713245.png)

有

![1564990746842](/img/in-post/2019-8-5-missing_data_VAE/1564990746842.png)

基于上述生成和识别模型，只考虑未缺失数据的边界分布 ELBO如下

![1564990962637](/img/in-post/2019-8-5-missing_data_VAE/1564990962637.png)

**估计缺失值的VAE可以表示为：**

![1564991475568](/img/in-post/2019-8-5-missing_data_VAE/1564991475568.png)

缺失数据识别模型不依赖于未缺失数据，即：

![1564992791879](/img/in-post/2019-8-5-missing_data_VAE/1564992791879.png)

### 处理异质（heteogenous）数据

在这里，我们考虑了第1节中介绍的数值和名义数据类型，我们提出以下似然模型

1. 实数集合数据

    对于实数集合数据，我们假设服从高斯分布函数，即：
    
    ![1564993176854](/img/in-post/2019-8-5-missing_data_VAE/1564993176854.png)
    
2. 正整数集合数据

    ![1564993490974](/img/in-post/2019-8-5-missing_data_VAE/1564993490974.png)

3. 离散计数数据

    ![1564993543850](/img/in-post/2019-8-5-missing_data_VAE/1564993543850.png)

4. 分类数据

    对于这一类数据，我们使用onehot进行编码，我们构建一个R维输出的DNN网络$$\gamma_{nd}=\{h_{d0}(z_n),h_{d1}(z_n),...,h_{dR-1}(z_n)\}$$，代表每一维概率，那么

    ![1564993936231](/img/in-post/2019-8-5-missing_data_VAE/1564993936231.png)

5. 序数数据

    这一类数据，使用刻度编码（thermometer encoding),

    ![1564994013196](/img/in-post/2019-8-5-missing_data_VAE/1564994013196.png)

## The Heterogeneous-Incomplete VAE (HI-VAE)

为了防止KL项主导ELBO，对$$z_n$$后验分布惩罚过多，我们可以通过$$z_n$$的先验分布在$$z_n$$中加强表示。

提出高斯混合分布（Gaussian mixture prior）$$p(z_n)$$：

![1565058601808](/img/in-post/2019-8-5-missing_data_VAE/1565058601808.png)

$$s_n$$是一个onehot编码的向量，代表混合的组成，即生成$z_n$的均值和方差信息。简单的说均一高斯混合对于所有$$\mathcal{l}$$都有$$\pi_\mathcal{l}=1/L$$,此外，为了简化模型准确捕获异构属性之间的统计依赖关系，我们提出了一种层次结构，允许不同的属性共享网络参数（即，分摊生成模型）.

我们引入中间同质数据表示$$Y=[y_{n1},y_{n2},...,y_{nD}]$$，是通过DNN网络和输入$$z_n$$，$$g(z_n)$$共同生成的。

![1564997396254](/img/in-post/2019-8-5-missing_data_VAE/1564997396254.png)

