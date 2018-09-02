---
layout:     post
title:      "reinforcement公式整理"
subtitle:   "烧锅炉烧累了，整理一波"
date:       2018-9-2
author:     "LCY"
header-img: "img/default.jpg"
tags:
    - python
    - reinforcement
    - DQN
    - policy gradient
    - tensorflow
---

## Monte Carlo Methods

有策略迭代如下：
$$
\pi_0\stackrel{E}{\rightarrow}q_{\pi_0}\stackrel{I}{\rightarrow} \pi_1\stackrel{E}{\rightarrow}q_{\pi_1}\stackrel{I}{\rightarrow} \cdot\cdot\cdot \stackrel{I}{\rightarrow}\pi_*\stackrel{E}{\rightarrow}q_{\pi_*}
$$

$$\mathit{E}$$: 策略评估(policy evaluation)，即利用样本估计行动值

$$\mathit{I}$$: 策略提升(policy improvement), 按照$$\pi_{k+1}(s) = {\arg\max}_a q_k(s,a)$$更新


![img](/img/in-post/reinforcement/1.jpg)

### 探索(exploration)

- 在线(on-policy): 更新值函数时使用当前策略产生的样本
  - policy is soft: 
    $$\pi(a|s)>0, s\in S, a \in A(s)$$
- 离线(off-policy): 更新值函数时不使用当前策略产生的样本

- $$\epsilon-greedy$$
  - 随机动作概率 
    $$\frac{\epsilon}{|A(s)|}$$
  - 贪心动作概率
    $$1-\epsilon+\frac{\epsilon}{|A(s)|}$$

#### 证明关于$$q_\pi$$的新$$\epsilon-greeedy$$策略$$\pi'$$优于原策略$$\pi$$
$$
\begin{align}
q_\pi(s, \pi'(s)) & = \sum_a\pi'(a|s)q_\pi(s,a) \\
& =\frac{\epsilon}{|A(s)|}\sum_a q_\pi(s,a)+(1-\epsilon)\max_a q_\pi(s,a) \\
& \geq \frac{\epsilon}{|A(s)|}\sum_a q_\pi(s,a)+(1-\epsilon)\sum_a\frac{\pi(a|s)-\frac{\epsilon}{|A(s)|}}{1-\epsilon}q_\pi(s,a) \\
& =\frac{\epsilon}{|A(s)|}\sum_a q_\pi(s,a)-\frac{\epsilon}{|A(s)|}\sum_a q_\pi(s,a)+\sum_a\pi(a|s)q_\pi(s,a) \\
& =v_\pi(s)
\end{align}
$$

当且仅当$$ \pi $$和$$ \pi' $$均为最优策略时，等号成立。

## DQN

$$
L(w) = \mathbb{E}[(r+\gamma \max_{\alpha'}Q(s',a',w)-Q(s,a,w))^2]
$$

![img](/img/in-post/reinforcement/2.jpg)

### Improve

#### Target Qnetwork

$$
I = (r+\gamma \max_{\alpha'}Q(s',a',w^-)-Q(s,a,w))^2
$$

两个网络，延迟更新

更新网络：

$$
w = \tau w^- + (1-\tau)w
$$

## policy gradient

$$
L(\theta) = \mathbb E(r_1+\gamma r_2 + \gamma^2 r_3 + ...|\pi(,\theta))
$$

更新参数$$\theta$$, 即对损失函数求导:

$$
\nabla_{\theta} L(\theta)
$$

仅仅从概率的角度来思考问题。我们有一个策略网络，输入状态，输出动作的概率。然后执行完动作之后，我们可以得到reward，或者result。**如果某一个动作得到reward多，那么我们就使其出现的概率增大，如果某一个动作得到的reward少，那么我们就使其出现的概率减小。**

**构造一个好的动作评判指标，来判断一个动作的好与坏，通过改变动作的出现概率来优化策略**

令这个评价指标为$$f(s,a)$$

则:

$$
L(\theta) = \sum log\pi(a|s,\theta)f(s,a)
$$

> #### [Why we consider log likelihood instead of Likelihood in Gaussian Distribution](https://math.stackexchange.com/questions/892832/why-we-consider-log-likelihood-instead-of-likelihood-in-gaussian-distribution)
>
> 1. It is extremely useful for example when you want to calculate the *joint likelihood* for a set of independent and identically distributed points. Assuming that you have your points:
> 
>    $$
>    X=\{x_1,x_2,\ldots,x_N\}
>    $$
> 
>    The total likelihood is the product of the likelihood for each point, i.e.:
> 
>    $$
>    p(X\mid\Theta)=\prod_{i=1}^Np(x_i\mid\Theta)
>    $$
> 
>    where$$ Θ$$ are the model parameters: vector of means $$ μ $$ and covariance matrix $$Σ$$ . If you use the log-likelihood you will end up with sum instead of product:
> 
>    $$
>    \ln p(X\mid\Theta)=\sum_{i=1}^N\ln
>    p(x_i\mid\Theta)
>    $$
>
> 2. Also in the case of Gaussian, it allows you to avoid computation of the exponential:
>
>    $$
>    p(x\mid\Theta) =
>    \dfrac{1}{(\sqrt{2\pi})^d\sqrt{\det\Sigma}}e^{-\frac{1}{2}(x-\mu)^T \Sigma^{-1}(x-\mu)}
>    $$
> 
>    Which becomes:
> 
>    $$
>    \ln p(x\mid\Theta) = -\frac{d}{2}\ln(2\pi)-\frac{1}{2}\ln(\det
>    \Sigma)-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)
>    $$
>
> 3. Like you mentioned lnx is a monotonically increasing function, thus log-likelihoods have the same relations of order as the likelihoods:
> 
>    $$
>    p(x\mid\Theta_1)>p(x\mid\Theta_2) \Leftrightarrow \ln
>    p(x\mid\Theta_1)>\ln p(x\mid\Theta_2)
>    $$
>
> 4. From a standpoint of computational complexity, you can imagine that first of all summing is less expensive than multiplication (although nowadays these are almost equal). But what is even more important, likelihoods would become very small and you will run out of your floating point precision very quickly, yielding an underflow. That's why it is way more convenient to use the logarithm of the likelihood. Simply try to calculate the likelihood by hand, using pocket calculator - almost impossible.
>
>    Additionally in the classification framework you can simplify calculations even further. The relations of order will remain valid if you drop the division by 2 and the $$dln(2π)$$ term. You can do that because these are class independent. Also, as one might notice if variance of both classes is the same ($$Σ_1=Σ_2$$), then you can also remove the $$ln(detΣ)$$ term.
>

$$
\begin{align}
\nabla_{\theta} E_x[f(x)] &= \nabla_{\theta} \sum_x p(x) f(x) & \text{definition of expectation} \\
& = \sum_x \nabla_{\theta} p(x) f(x) & \text{swap sum and gradient} \\
& = \sum_x p(x) \frac{\nabla_{\theta} p(x)}{p(x)} f(x) & \text{both multiply and divide by } p(x) \\
& = \sum_x p(x) \nabla_{\theta} \log p(x) f(x) & \text{use the fact that } \nabla_{\theta} \log(z) = \frac{1}{z} \nabla_{\theta} z \\
& = E_x[f(x) \nabla_{\theta} \log p(x) ] & \text{definition of expectation}
\end{align}
$$

[ 策略梯度方法](https://zhuanlan.zhihu.com/p/26174099)



## Actor-Critic

[Actor-Critic算法小结](https://zhuanlan.zhihu.com/p/29486661)

## Primal-Dual DDPG

#### MDP

马尔科夫决策过程(markov Decision Process), 看作一个元组$$(\mathcal{S}, \mathcal{A}, \mathcal{R}, \mathcal{P}, p_0)$$

$$\mathcal{R}$$: **状态state**: $$\mathcal{R}$$

$$\mathcal{R}$$: **奖赏reward**: $$\mathcal{R}:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\mapsto \mathbb{R}$$

$$\mathcal{P}$$: **传播概率transition probability**
$$\mathcal{P}:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\mapsto [0,1]$$ (where $$P(s'|s,a)$$ is the transition probability from state $$s$$ to state $$s'$$ given action $$a$$).

$$p_0$$: 初始状态分布 $$\mathcal{S}\mapsto [0,1]$$ 


静态策略$$\pi$$,映射了状态集合到动作集合的概率分布, 如$$\pi(\alpha|s)$$是在状态$$s$$时选择动作$$\alpha$$的概率.

在策略$$\pi$$下的长期奖赏

$$
R(\pi)=\mathbb{E}_{\tau \thicksim \pi}[\sum_{t=0}^{\infty}\gamma^t R(s_t,a_t,s_{t+1})]
$$

其中$$\gamma \in [0,1)$$,是折扣因子（discount factor）。$$\tau=(s_0,a_0,s_1,a_1,...)$$是决策路径（trajectory）。其中$$\tau \thicksim \pi$$意味着决策路径下的分布由策略$$\pi$$所确定,如

$$
s_0 \thicksim p_0,\ a_t \thicksim \pi(.|s_t),\ s_{t+1} \thicksim P(.|s_t,a_t)
$$

### CMDP

受限马尔科夫决策过程（constrained Markov Decision Procession）是使用长期折扣代价（discount count），补充了原马尔科夫决策理论。

在原有MDP上定义代价$$C_1,C_2, ...$$, 每个片段上的代价$$C_i:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\mapsto \mathbb{R}$$.长期代价就是$$C_i(\pi)=\mathbb{E}_{\tau \thicksim \pi}[\sum_{t=0}^\infty \gamma^tC_i(s_t,a_t,s_{t+1})]$$, 对应的限制是$$d_i$$

**我们的目标是，在满足 $$C_i(\pi) \le d_i,\forall i \in [m]$$的条件下,选出一个策略$$\pi$$使长期奖励$$R(\pi)$$最大。**

$$
\pi^*=arg\max_{\pi in \Pi_\theta}R(\pi)\\
s.t. C_i(\pi)\le d_i, \forall i \in [m]
$$

### algorithm
使用拉格朗日方法求解

$$
\mathcal{L}(\pi,\lambda)=R(\pi)=\sum \lambda_i(C_i(\pi)-d_i)
$$

其中$$\lambda=(\lambda_1,...,\lambda_m)$$是拉格朗日乘子。

$$
\pi^*=arg\max_{\pi in \Pi_\theta}R(\pi)\\
s.t. C_i(\pi)\le d_i, \forall i \in [m]
$$

可以看作：

$$
(\pi^*,\lambda^*)=arg\min_{\lambda \ge 0}\max_{\pi \in \Pi_\theta}\mathcal(\pi,\lambda)
$$

为解决无约束条件下最大最小问题，在每次迭代过程中一次更新策略$$\pi$$和$$\lambda$$

每次迭代：

1. 固定$$\lambda=\lambda^{(k)}$$，执行梯度策略更新：
   $$\theta_{k+1}=\theta_k + \alpha_k\nabla_{\theta}(\mathcal{L}(\pi(\theta),\lambda^{(k)}))|_{\theta=\theta_k}$$
3. 固定$$\pi=\pi_k$$，执行双重更新，$$\lambda^{(k+1)}=f_k(\lambda^{(k)},\pi_k)$$

![](/img/in-post/reinforcement/3.png)

# paper

[Accelerated Primal-Dual Policy Optimization for Safe Reinforcement Learning. Qingkai Liang, Fanyu Que, Eytan Modiano](https://arxiv.org/abs/1802.06480)