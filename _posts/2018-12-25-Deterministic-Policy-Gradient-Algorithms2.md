---
layout:     post
title:      "《Deterministic Policy Gradient Algorithms》阅读笔记（or 翻译）2"
subtitle:   "阅读笔记（or 翻译）"
date:       2018-12-25
author:     "LCY"
header-img: "img/default.jpg"
tags:
    - ML
    - reinforcement learning
    - paper
---
[TOC]

# 《Deterministic Policy Gradient Algorithms》阅读笔记（or 翻译）

## 原论文&附录

[Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)
[Deterministic Policy Gradient Algorithms: Supplementary Materia(论文附录)](http://proceedings.mlr.press/v32/silver14-supp.pdf)

## 论文摘要（Abstract） 

在本文中，我们考虑确定性策略梯度（Deterministic Policy Gradient）算法，用于连续行动的强化学习。确定性策略梯度具有特别吸引人的形式：它是动作 - 值函数的预期梯度。这种简单的形式意味着可以它可以比通常的随机政策梯度具有更有效率的估计。为了确保充分的探索，我们引入了一种离线（off-policy）actor-critic算法，该算法从探索性行为策略中学习确定性目标策略。我们证明了确定性策略梯度算法在高维动作空间中可以显着优于其随机对应部分。

## 简介（Introduction）

随机策略梯度算法广泛用于具有连续动作空间的强化学习问题。基本思想是通过在当前状态$$s$$, 根据参数向量$$\theta$$ 来随机的选取动作$$a$$ ，来确定一个策略$$\pi_\theta(a|s)=\mathbb P[a|s;\theta]$$ 。在这篇论文中，我们令确定性策略$$a=\mu_\theta(s)$$ 。

以前认为，确定性梯度策略不存在或只model-base的情况才会存在[Peters, J. (2010).  Policy gradient methods.Scholarpedia, 5(11):3698]。然而在这篇论文中，作者指出确定性梯度策略确实存在，并且它具有简单的遵循动作-值函数梯度的无模形式（model-free ），确定性梯度策略是策略差异趋向于0的随机梯度策略的特殊情况。

从实践的角度来看，随机策略和确定性策略梯度之间存在着至关重要的差异。在随机情况下，策略梯度在状态和动作空间上进行整合，而在确定性情况下，它仅在状态空间上进行整合。计算随机策略梯度可能需要更多样本，特别是在动作空间有许多维度的情况下。

## 背景知识（background）

增强学习是在时间序列上针对不同state选取action的过程，我们将之建模为马尔科夫过程（MDP），该过程包含：状态空间$$\mathcal{S}$$，动作空间$$\mathcal{A}$$，初始状态分布(概率密度函数)$$p_1(s_1)$$，动态转移概率$$p(s_{t+1}|s_t, a)$$（该概率满足马尔科夫的属性$$p(s_{t+1}|s_1,a_1,...,s_t,a_t)=p(s_{t+1}|s_t,a_t)$$），评价函数$$r:\mathcal{S}\times\mathcal{A}\to \mathbb{R}$$。

另$$\mathcal{P}(\mathcal{A})$$为$$\mathcal{A}$$的概率集合，$$\theta\in\mathbb{R}^n$$为一组$$n$$维向量，则选取action的策略为$$\pi_\theta:\mathcal{S}\to\mathcal{P}(\mathcal{A})$$ 。$$\pi_\theta(a_t|s_t)$$是当前策略转移到$$a_t$$的条件概率。agent用自己的策略与环境交互，产生状态-动作-奖励的MDP过程：$$h_{1:T}=s_1,a_1,r_1,...,s_T,a_T,r_T$$，其中$$h_{1:T}\in \mathcal{A}\times\mathcal{A}\times\mathbb{R}$$。$$t^\gamma_t$$是从$$t$$往前的总折扣奖励，即，当$$0<\gamma<1$$时，$$r_t^\gamma=\sum_{k=t}^\infty \gamma^{k-t}r(s_k,a_k)$$。而预期总奖励为$$V^\pi(s)=\mathbb{E}[r_1^\gamma|S_1=s;\pi]$$，$$Q^\pi(s,a)=\mathbb{E}[r_1^\gamma|S_1=s,A_1=a;\pi]$$。agent的目标为获取一个策略，使累积奖励最大化。目标函数为$$J(\pi)=\mathbb{E}[r_1^\gamma|\pi]$$。

从状态$$s$$经过$$t$$时间过度到状态$$s'$$的概率密度为$$p(s\to s',t,\pi)$$。状态分布为$$\rho^\pi(s'):=\int_{\mathcal{S}}\sum_{t=1}^\infty \gamma^{t-1}p_1(s)P(s \to s', t, \pi)ds$$。把目标函数写作期望的话：

$$
\begin{aligned}
J(\pi_\theta)&=\int_\mathcal{S}\rho^\pi(s)\int_\mathcal{A}\pi_\theta(s,a)r(s,a)dads \\
             &=\mathbb{E}_{\mathcal{S}\sim\rho^\pi,a\sim\pi_\theta}[r(s,a)]
\end{aligned}
$$


### 随机策略梯度定理（Stochastic Policy Gradient Theorem）

下面的推导主要依赖于$$\nabla \log f(x)=\frac{\nabla f(x)}{f(x)}$$这一性质。

$$
\begin{aligned}
\nabla_\theta J(\pi_\theta)&=\int_{\mathcal{S}}\rho^\pi(s)\int_\mathcal{A}\nabla_\theta\pi_\theta(a|s)Q^\pi(s,a)dads \\
             &=\int_{\mathcal{S}}\rho^\pi(s)\int_\mathcal{A}Q^\pi(s,a)\nabla_\theta\pi_\theta(a|s)dads \\
             &=\int_{\mathcal{S}}\rho^\pi(s)\int_\mathcal{A}Q^\pi(s,a)\pi_\theta(a|s)\frac{\nabla_\theta\pi_\theta(a|s)}{\pi_\theta(a|s)}dads        \\
             &=\int_{\mathcal{S}}\rho^\pi(s)\int_\mathcal{A}Q^\pi(s,a)\pi_\theta(a|s)\nabla\log \pi_\theta(a|s)dads \\
             &=\mathbb{E}_{\mathcal{S}\sim\rho^\pi,a\sim\pi_\theta}[Q^\pi(s,a)\nabla_\theta log\pi_\theta(a|s)] \\
             &=\mathbb{E}_{\mathcal{S}\sim\rho^\pi,a\sim\pi_\theta}[\nabla_\theta log\pi_\theta(a|s)Q^\pi(s,a)]
\end{aligned}
$$

综上，尽管状态分布$$\rho^\pi(s)$$依赖于策略的参数，策略梯度却不依赖与状态分布的梯度。

### 随机actor-critic算法（Stochastic Actor-Critic Algorithms）*

基于梯度策略理论的actor-critic算法是一个被广泛使用的框架(Sutton et al., 1999; Peters et al.,2005;  Bhatnagar  et  al.,  2007;  Degris  et  al.,  2012a)。该框架主要由两部分组成，actor通过随机梯度上升用来更新随机策略$$\pi_\theta(s)$$的参数$$\theta$$。虽然我们不知道公式中的真实action-value对应关系的函数$$Q^\pi(s,a)$$，但是我们使用参数$$w$$创建近似的价值函数$$Q^w(s,a)$$，通过合适的策略计算算法，可以尽量令$$Q^w(s,a)\simeq Q^\pi(s,a)$$。

通常来说，用$$Q^w(s,a)$$来逼近$$Q^\pi(s,a)$$会引入偏差，为了消除偏差，应满足：

1. $$Q^w(s,a)=\nabla_\theta\log\pi_\theta(a|s)^Tw$$（对于随机策略的函数逼近为线性）
2. 参数$$w$$，应该最小化均方误差$$\varepsilon^2(w)=\mathbb{E}_{\mathcal{S}\sim\rho^\pi,a\sim\pi_\theta}[(Q^w(s,a)-Q^\pi(s,a))^2]$$

那么我们的策略梯度为：

$$
\nabla_\theta J(\pi_\theta)=\mathbb{E}_{\mathcal{S}\sim\rho^\pi,a\sim\pi_\theta}[\nabla_\theta log\pi_\theta(a|s)Q^w(s,a)]
$$

在实践中，放宽条件2，更有利于算法通过时间差分学习到更有效的评估函数。如果条件1，2全部满足的话，整个算法相当于不需要使用critic。

### 离线actor-critic学习（Off-Policy Actor-Critic）*

从轨迹中不同的策略行为$$\beta(a|s) \neq \pi_\theta(a|s)$$，离线估计策略梯度是很常用的方法。在离线环境下，评估目标通常修改为目标策略的价值函数。

$$
\begin{aligned}
J_\beta(\pi_\theta)&=\int_\mathcal{S}\rho^{\beta}(s)V^\pi(s)ds \\
                   &=\int_\mathcal{S}\int_\mathcal{A}\rho^\beta(s)\pi_\theta(a|s)Q^\pi(s,a)dads
\end{aligned}
$$

梯度约为

$$
\begin{aligned}
\nabla_\theta J_\beta(\pi_\theta) &\approx \int_\mathcal{S}\int_\mathcal{A}\rho^\beta(s)\nabla_\theta\pi_\theta(a|s)Q^\pi(s,a)dads \\
                                  &= \mathbb{E}_{\mathcal{S}\sim\rho^\beta,a\sim\beta}[\frac{\pi_\theta(a|s)}{\beta_\theta(a|s)}\nabla_\theta \log \pi_\theta(a|s)Q^\pi(s,a)]
\end{aligned}
$$

离线actor-critic算法（OffPAC）使用行为策略（behaviour  policy）$$\beta(a|s)$$来生成轨迹。critic是一个状态-价值的函数$$V^v(s)\approx V^\pi(s)$$。actor用来更新策略的参数$$\theta$$，actor和critc都是通过离线的轨迹数据进行训练。和公式中$$Q^\pi(s,a)$$不同的是，我们使用时间差分误差$$\delta_t=r_{t+1}+\gamma V^v(s_{t+1})-V^v(s_t)$$，这样可以提供真实梯度的近似值。可以使用$$\frac{\pi_\theta(a|s)}{\beta_\theta(a|s)}$$这一比率判断action到底是根据策略$$\pi$$还是$$\beta$$。

## 

## 确定性策略梯度（Gradients of Deterministic Policies）

现在考虑如何将策略梯度框架扩展到确定性策略。类似于前面部分提出的随机政策梯度定理。确定性策略梯度实际上是随机策略梯度的一个特例。

### 动作-值函数梯度（Action-Value Gradients）

大多数model-free强化学习都是基于一般的策略迭代：策略评估和策略更新交错进行(Sutton and Barto, 1998)。策略评估是通过蒙特卡洛或时间差分算法产生数据，拟合$$Q^\pi(s,a)$$，$$Q^\mu(s,a)$$，最常见的算法是采用贪心算法，最大化动作-值函数：$$\mu^{k+1}(s)=\underset{a}{\operatorname{argmax}} Q^{\mu^k}(s,a)$$。

但是在动作空间连续的情况下，贪心算法需要在每一步都最大化，就产生了问题。一个简单的替代方案是将策略往$$Q$$的梯度方向移动，而不是全局最大化$$Q$$。具体来说，对于每一个探索过的状态$$s$$，策略网络的参数$$\theta^{k+1}$$以$$\nabla_\theta Q^{\mu^k}(s, \mu_\theta(s))$$的一定比例来更新。每个不同的状态，都提供了一个更新的方向，所有方向的均值，可以看作$$\rho^\mu(s)$$。

$$
\theta^{k+1}:=\theta^k+\alpha\mathbb{E}_{\mathcal{S}\sim\rho^{\mu^k}}[\nabla_\theta Q^{\mu^k}(s,\mu_\theta(s))]
$$

策略更新可以分解为动作-值函数的梯度和评估策略的梯度更新

$$
\theta^{k+1}:=\theta^k+\alpha\mathbb{E}_{\mathcal{S}\sim\rho^{\mu^k}}[\nabla_\theta Q^{\mu^k}(s,a)|_{a=\mu_\theta(s)}]
$$

按照惯例，$$\nabla_\theta\mu_\theta(s)$$是一个雅可比矩阵，也就是说，每一列都是梯度$$\nabla_\theta[\mu_\theta(s)]_d$$（$$d$$是动作空间的维度）。通过改变策略，不同的状态都会被探索，并且状态分布$$\rho^\mu$$也会被改变。

### 确定性策略梯度定理（Deterministic Policy Gradient Theorem）

现在考虑带有参数向量$$\theta\in\mathbb{R}^n$$确定性策略$$\mu_\theta:\mathcal{S}\to\mathcal{A}$$。定义目标函数为$$J(\mu_\theta)=\mathbb{E}[r^\gamma_1|\mu]$$，定义概率分布$$p(s\to s',t,\mu)$$，折扣状态分布$$\rho^\mu(s)$$，将目标函数写为期望：

$$
\begin{aligned}
J(\mu_\theta)&=\int_{\mathcal{S}}\rho^\mu(s)r(s,\mu_\theta(s))ds \\
             &=\mathbb{E}_{\mathcal{S}\sim\rho^\mu}[r(s,\mu_\theta(s))]
\end{aligned}
$$


#### 定理1(Deterministic  Policy  Gradient  Theorem)

如果MDP过程满足$$p(s'|s, a)$$，$$\nabla_a p(s'|s, a)$$，$$\mu_\theta(s)$$，$$\nabla_\theta\mu_\theta(s)$$，$$r(s,a)$$，$$\nabla_a r(s,a)$$，$$p_1(s)$$，在参数$$s$$, $$a$$, $$s'$$, $$x$$下都是连续的，那么意味着$$\nabla_\theta\mu_\theta(s)$$和$$\nabla_aQ^\mu(s,a)$$存在且确定性策略梯度存在。那么：

$$
\begin{aligned}
\nabla_\theta J(\mu_\theta)&=\int_{\mathcal{S}}\rho^\mu(s)\nabla_\theta\mu_\theta(s)\nabla_aQ^\mu(s,a)|_{a=\mu_\theta(s)}ds \\
             &=\mathbb{E}_{\mathcal{S}\sim\rho^\mu}[\nabla_\theta\mu_\theta(s)\nabla_aQ^\mu(s,a)|_{a=\mu_\theta(s)}]
\end{aligned}
$$


##### 证明

如果MDP满足$$p(s'|s, a)$$，$$\nabla_a p(s'|s, a)$$，$$\mu_\theta(s)$$，$$\nabla_\theta\mu_\theta(s)$$，$$r(s,a)$$，$$\nabla_a r(s,a)$$，$$p_1(s)$$，在参数$$s$$, $$a$$, $$s'$$, $$x$$下都是连续的，那么$$V^{\mu_\theta}(s)$$和$$\nabla_\theta V^{\mu_\theta}(s)$$对于$$\theta$$和$$s$$都是连续的。对于任意$$\theta$$，状态空间$$\mathcal{S}$$是紧凑的，因此$$||\nabla_\theta V^{\mu_\theta}{(s)}||$$，$$||\nabla_a Q^{\mu_\theta}(s,a)|_{a=\mu_\theta(s)}||$$，$$\nabla_\theta\mu_\theta(s)$$都是$$s$$的有界函数。

$$
\begin{aligned}
\nabla_\theta V^{\mu_\theta}(s) &= \nabla_\theta Q^{\mu_\theta}(s,\mu_\theta(s)) \\
                                &= \nabla_\theta\lgroup r(s,\mu_\theta(s))+\int_{\mathcal{S}}\gamma p(s'|s,\mu_\theta(s))V^{\mu_\theta}(s')ds' \rgroup \\
                                &= \nabla_\theta\mu_\theta(s)\nabla_a r(s,a)|_{a=\mu_\theta(s)} + \nabla_\theta\int_{\mathcal{S}}\gamma p(s'|s,\mu_\theta(s))V^{\mu_\theta}(s')ds' \\
                                &= \nabla_\theta\mu_\theta(s)\nabla_a r(s,a)|_{a=\mu_\theta(s)} \\
                 & \ \ \ \ \ \ + \int_\mathcal{S}\gamma\lgroup p(s'|s,\mu_\theta(s))\nabla_\theta V^{\mu_\theta}(s')+\nabla_\theta\mu_\theta(s)\nabla_ap(s'|s,a)|_{a=\mu_\theta(s)}V^{\mu_\theta}(s') \rgroup ds' \\
                                &= \nabla_\theta\mu_\theta(s)\nabla_a \lgroup r(s,a)+ \int_\mathcal{S}\gamma p(s'|s,a)V^{\mu_\theta}(s')ds' \rgroup |_{a=\mu_\theta(s)} \\
                 & \ \ \ \ \ \ + \int_\mathcal{S}\gamma p(s'|s,\mu_\theta(s))\nabla_\theta V^{\mu_\theta}(s')ds' \\
                                &= \nabla_\theta\mu_\theta(s)\nabla_a Q^{\mu_\theta}(s,a)|_{a=\mu_\theta(s)}+\int_\mathcal{S}\gamma p(s\to s',1,\mu_\theta)\nabla_\theta V^{\mu_\theta}(s')ds'
\end{aligned}
$$

迭代这一过程消去$$V^{\mu_\theta}$$:

$$
\begin{aligned}
\nabla_\theta V^{\mu_\theta}(s) &= \nabla_\theta\mu_\theta(s)\nabla_a Q^{\mu_\theta}(s,a)|_{a=\mu_\theta(s)} + \int_\mathcal{S}\gamma p(s\to s',1,\mu_\theta)\nabla_\theta V^{\mu_\theta}(s')ds' \\
    &= \nabla_\theta\mu_\theta(s)\nabla_a Q^{\mu_\theta}(s,a)|_{a=\mu_\theta(s)} \\
    &\ \ \ \ \ \ + \int_\mathcal{S}\gamma p(s\to s',1,\mu_\theta)\nabla_\theta \mu_\theta(s')Q^{\mu_\theta}(s',a)ds' \\
    &\ \ \ \ \ \ + \int_\mathcal{S}\gamma^2 p(s\to s',2,\mu_\theta)\nabla_\theta \mu_\theta(s')Q^{\mu_\theta}(s',a)ds' \\
    &\ \ \ \ \ \ + \ ... \\
    &=\int_\mathcal{S}\sum^\infty_{t=0}\gamma^t p(s\to s',t,\mu_\theta)\nabla_\theta\mu_\theta(s')\nabla_a Q^{\mu_\theta}(s',a)|_{a=\mu_\theta(s')}ds'
\end{aligned}
$$

整个推导过程中，有几块运用到了梯度的链式法则：

$$
\begin{aligned}
\frac{df(a,g(x))}{dx}&=\frac{df(a,y)}{dx}|_{y=g(x)} \\
                     &=\frac{dy}{dx} \cdot \frac{df(a,y)}{dy}|_{y=g(x)} \\
                     &=\frac{dg(x)}{dx} \cdot \frac{df(a,y)}{dy}|_{y=g(x)}
\end{aligned}
$$

因此，我们有：

$$
\begin{aligned}
\nabla_\theta J(\mu_\theta)&=\nabla_\theta\int_\mathcal{S}p_1(s)V^{\mu_\theta}(s)ds \\
    &=\int_\mathcal{S}p_1(s)\nabla_\theta V^{\mu_\theta}(s)ds \\
    &=\int_\mathcal{S}\int_\mathcal{S}\sum_{t=0}^\infty \gamma^t p_1(s)p(s\to s',t,\mu_\theta)\nabla_\theta\mu_\theta(s')\nabla_a Q^{\mu_\theta}(s',a)|_{a=\mu_\theta(s')}ds'ds \\
    &=\int_\mathcal{S} \rho^{\mu_\theta}(s)\nabla_\theta\mu_\theta(s)\nabla_a Q^{\mu_\theta}(s,a)|_{a=\mu_\theta(s)}ds
\end{aligned}
$$


#### 确定性策略是随机策略梯度的特殊情况

我们把随机策略$$\pi_{\mu_\theta,\sigma}$$用确定策略$$\mu_\theta:\mathcal{S}\to \mathcal{A}$$和方差$$\sigma$$ 来代替，当$$\sigma=0$$时，$$\pi_{\mu_\theta,0}\equiv \mu_\theta$$，随机策略等同于确定性策略。

$$
\lim_{\sigma \downarrow 0}\nabla_\theta J(\pi_{\mu_\theta})=\nabla_\theta J(\mu_\theta)
$$

(下箭头是指$$\sigma$$单调递减收敛于0)

##### 证明*

// TODO

（http://proceedings.mlr.press/v32/silver14-supp.pdf）

### 确定性actor-critic算法（Deterministic Actor-Critic Algorithms）

#### 在线确定性actor-critic（On-Policy Deterministic Actor-Critic）

critic用来评估动作-值函数，actor根据动作-值函数进行梯度上升。actor根据以下公式调整参数$$\theta$$

$$
\begin{aligned}
\nabla_\theta J(\mu_\theta)&=\int_{\mathcal{S}}\rho^\mu(s)\nabla_\theta\mu_\theta(s)\nabla_aQ^\mu(s,a)|_{a=\mu_\theta(s)}ds \\
             &=\mathbb{E}_{\mathcal{S}\sim\rho^\mu}[\nabla_\theta\mu_\theta(s)\nabla_aQ^\mu(s,a)|_{a=\mu_\theta(s)}]
\end{aligned}
$$

与随机策略梯度一样，我们用可微分的动作-值函数$$Q^w(s,a)$$来代替真实的动作-值函数$$Q^\mu(s,a)$$。critic的作用就是采用适当的迭代算法使$$Q^w(s,a)\simeq Q^\mu(s,a)$$。例如，采用Sarsa更新的话：

$$
\begin{aligned}
\sigma_t &= r_t+\gamma Q^w(s_{t+1)},a_{(t+1)})-Q^w(s_t,a_t) \\
w_{t+1} &= w_t+\alpha_w\sigma_t\nabla_w Q^w(s_t,a_t) \\
\theta_{t+1} &=\theta_t+\alpha_\theta\nabla_\theta\mu_\theta(s_t)\nabla_a Q^w(s_t,a_t)|_{a=\mu_\theta(s)}
\end{aligned}
$$


#### 离线确定性策略（Off-Policy Deterministic Actor-Critic）

现在考虑离线策略，$$\mu_\theta(s)$$从一系列轨迹数据中学习。我们的目标函数为：

$$
\begin{aligned}
J_\beta(\mu_\theta)&=\int_\mathcal{S}\rho^\beta(s)V^\mu(s)ds \\
  &= \int_\mathcal{S}\rho^\beta(s)Q^\mu(s,\mu_\theta(s))ds 
\end{aligned}
$$

梯度为：

$$
\begin{aligned}
\nabla_\theta J\beta(\mu_\theta)&\simeq \int_\mathcal{S}\rho^\beta(s)\nabla_\theta\mu_\theta(a|s)Q^\mu(s,a)ds \\
  &= \mathbb{E}_{\mathcal{S}\sim \rho^\beta}[\nabla_\theta\mu_\theta(s)\nabla_aQ\mu(s,a)|_{a=\mu_\theta(s)}]
\end{aligned}
$$

同样的，用可微分的动作-值函数$$Q^w(s,a)$$来代替真实的动作-值函数$$Q^\mu(s,a)$$。离线学习通过$$\beta(a|s)$$产生轨迹数据。下面是ciritc通过Q-learning的学习策略进行更新：

$$
\begin{aligned}
\sigma_t &= r_t+\gamma Q^w(s_{t+1},\mu_\theta(s_{t+1}))-Q^w(s_t,a_t) \\
w_{t+1}  &= w_t+\alpha_w\sigma_t\nabla_wQ^w(s_t,a_t) \\
\theta_{t+1}&=\theta_t+\alpha_\theta\nabla_\theta\mu_\theta(s_t)\nabla_aQ^w(s_t,a_t)|_{s=\mu_\theta(s)}
\end{aligned}
$$


#### Compatible Function Approximation

用近似的$$Q^w$$代替真实情况，不一定代表策略梯度是真实情况的梯度。

如果近似函数$$Q^w(s,a)$$与$$\mu_\theta(s)$$，$$\nabla_\theta J_\beta(\mu^\theta)=\mathbb{E}_{\mathcal{S}\sim\rho^\beta}[\nabla_\theta\mu_\theta(s)\nabla_aQ^\mu(s,a)|_{a=\mu_\theta}]$$兼容，那么满足以下两个条件：

- $$\nabla_aQ^w(s,a)|_{a=\mu_\theta(s)}=\nabla_\theta\mu_\theta(s)^Tw$$
- $$w$$y应最小化均方误差$$\mathrm{MSE}(\theta,w)=\mathbb{E}[\epsilon(s;\theta,w)^T\epsilon(s;\theta,w)]$$，其中，$$\epsilon(s;\theta,w)=\nabla_aQ^w(s,a)|_{a=\mu_\theta(s)}-\nabla_aQ^\mu(s,a)|_{a=\mu_\theta(s)}$$

##### 证明*

如果$$w$$最小化 MSE，那么$$\epsilon^2$$对$$w$$ 的梯度一定是0。根据条件1，$$\nabla_w\epsilon(s;\theta,w)=\nabla_\theta\mu_\theta(s)$$。

$$
\nabla_w\mathrm{MSE}(\theta,w)=0 \\
\mathbb{E}[\nabla_\theta\mu_\theta(s)\epsilon(s;\theta,w)]=0 \\
\mathbb{E}[\nabla_\theta\mu_\theta(s)\nabla_aQ^w(s,a)|_{a=\mu_\theta(s)}]=\mathbb{E}[\nabla_\theta\mu_\theta(s)\nabla_aQ^\mu(s,a)|_{a=\mu_\theta(s)}] \\
=\nabla_\theta J_\beta(\mu_\theta) \ \mathrm{or}\ \nabla_\theta J(\mu_\theta)
$$

//TODO: