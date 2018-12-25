---
layout:     post
title:      "��Deterministic Policy Gradient Algorithms���Ķ��ʼǣ�or ���룩2"
subtitle:   "�Ķ��ʼǣ�or ���룩"
date:       2018-12-25
author:     "LCY"
header-img: "img/default.jpg"
tags:
    - ML
    - reinforcement learning
    - paper
---
[TOC]

# ��Deterministic Policy Gradient Algorithms���Ķ��ʼǣ�or ���룩

## ԭ����&��¼

[Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)
[Deterministic Policy Gradient Algorithms: Supplementary Materia(���ĸ�¼)](http://proceedings.mlr.press/v32/silver14-supp.pdf)

## ����ժҪ��Abstract�� 

�ڱ����У����ǿ���ȷ���Բ����ݶȣ�Deterministic Policy Gradient���㷨�����������ж���ǿ��ѧϰ��ȷ���Բ����ݶȾ����ر������˵���ʽ�����Ƕ��� - ֵ������Ԥ���ݶȡ����ּ򵥵���ʽ��ζ�ſ��������Ա�ͨ������������ݶȾ��и���Ч�ʵĹ��ơ�Ϊ��ȷ����ֵ�̽��������������һ�����ߣ�off-policy��actor-critic�㷨�����㷨��̽������Ϊ������ѧϰȷ����Ŀ����ԡ�����֤����ȷ���Բ����ݶ��㷨�ڸ�ά�����ռ��п������������������Ӧ���֡�

## ��飨Introduction��

��������ݶ��㷨�㷺���ھ������������ռ��ǿ��ѧϰ���⡣����˼����ͨ���ڵ�ǰ״̬$$s$$, ���ݲ�������$$\theta$$ �������ѡȡ����$$a$$ ����ȷ��һ������$$\pi_\theta(a|s)=\mathbb P[a|s;\theta]$$ ������ƪ�����У�������ȷ���Բ���$$a=\mu_\theta(s)$$ ��

��ǰ��Ϊ��ȷ�����ݶȲ��Բ����ڻ�ֻmodel-base������Ż����[Peters, J. (2010).  Policy gradient methods.Scholarpedia, 5(11):3698]��Ȼ������ƪ�����У�����ָ��ȷ�����ݶȲ���ȷʵ���ڣ����������м򵥵���ѭ����-ֵ�����ݶȵ���ģ��ʽ��model-free ����ȷ�����ݶȲ����ǲ��Բ���������0������ݶȲ��Ե����������

��ʵ���ĽǶ�������������Ժ�ȷ���Բ����ݶ�֮�������������Ҫ�Ĳ��졣���������£������ݶ���״̬�Ͷ����ռ��Ͻ������ϣ�����ȷ��������£�������״̬�ռ��Ͻ������ϡ�������������ݶȿ�����Ҫ�����������ر����ڶ����ռ������ά�ȵ�����¡�

## ����֪ʶ��background��

��ǿѧϰ����ʱ����������Բ�ͬstateѡȡaction�Ĺ��̣����ǽ�֮��ģΪ����Ʒ���̣�MDP�����ù��̰�����״̬�ռ�$$\mathcal{S}$$�������ռ�$$\mathcal{A}$$����ʼ״̬�ֲ�(�����ܶȺ���)$$p_1(s_1)$$����̬ת�Ƹ���$$p(s_{t+1}|s_t, a)$$���ø�����������Ʒ������$$p(s_{t+1}|s_1,a_1,...,s_t,a_t)=p(s_{t+1}|s_t,a_t)$$�������ۺ���$$r:\mathcal{S}\times\mathcal{A}\to \mathbb{R}$$��

��$$\mathcal{P}(\mathcal{A})$$Ϊ$$\mathcal{A}$$�ĸ��ʼ��ϣ�$$\theta\in\mathbb{R}^n$$Ϊһ��$$n$$ά��������ѡȡaction�Ĳ���Ϊ$$\pi_\theta:\mathcal{S}\to\mathcal{P}(\mathcal{A})$$ ��$$\pi_\theta(a_t|s_t)$$�ǵ�ǰ����ת�Ƶ�$$a_t$$���������ʡ�agent���Լ��Ĳ����뻷������������״̬-����-������MDP���̣�$$h_{1:T}=s_1,a_1,r_1,...,s_T,a_T,r_T$$������$$h_{1:T}\in \mathcal{A}\times\mathcal{A}\times\mathbb{R}$$��$$t^\gamma_t$$�Ǵ�$$t$$��ǰ�����ۿ۽�����������$$0<\gamma<1$$ʱ��$$r_t^\gamma=\sum_{k=t}^\infty \gamma^{k-t}r(s_k,a_k)$$����Ԥ���ܽ���Ϊ$$V^\pi(s)=\mathbb{E}[r_1^\gamma|S_1=s;\pi]$$��$$Q^\pi(s,a)=\mathbb{E}[r_1^\gamma|S_1=s,A_1=a;\pi]$$��agent��Ŀ��Ϊ��ȡһ�����ԣ�ʹ�ۻ�������󻯡�Ŀ�꺯��Ϊ$$J(\pi)=\mathbb{E}[r_1^\gamma|\pi]$$��

��״̬$$s$$����$$t$$ʱ����ȵ�״̬$$s'$$�ĸ����ܶ�Ϊ$$p(s\to s',t,\pi)$$��״̬�ֲ�Ϊ$$\rho^\pi(s'):=\int_{\mathcal{S}}\sum_{t=1}^\infty \gamma^{t-1}p_1(s)P(s \to s', t, \pi)ds$$����Ŀ�꺯��д�������Ļ���

$$
\begin{aligned}
J(\pi_\theta)&=\int_\mathcal{S}\rho^\pi(s)\int_\mathcal{A}\pi_\theta(s,a)r(s,a)dads \\
             &=\mathbb{E}_{\mathcal{S}\sim\rho^\pi,a\sim\pi_\theta}[r(s,a)]
\end{aligned}
$$


### ��������ݶȶ���Stochastic Policy Gradient Theorem��

������Ƶ���Ҫ������$$\nabla \log f(x)=\frac{\nabla f(x)}{f(x)}$$��һ���ʡ�

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

���ϣ�����״̬�ֲ�$$\rho^\pi(s)$$�����ڲ��ԵĲ����������ݶ�ȴ��������״̬�ֲ����ݶȡ�

### ���actor-critic�㷨��Stochastic Actor-Critic Algorithms��*

�����ݶȲ������۵�actor-critic�㷨��һ�����㷺ʹ�õĿ��(Sutton et al., 1999; Peters et al.,2005;  Bhatnagar  et  al.,  2007;  Degris  et  al.,  2012a)���ÿ����Ҫ����������ɣ�actorͨ������ݶ��������������������$$\pi_\theta(s)$$�Ĳ���$$\theta$$����Ȼ���ǲ�֪����ʽ�е���ʵaction-value��Ӧ��ϵ�ĺ���$$Q^\pi(s,a)$$����������ʹ�ò���$$w$$�������Ƶļ�ֵ����$$Q^w(s,a)$$��ͨ�����ʵĲ��Լ����㷨�����Ծ�����$$Q^w(s,a)\simeq Q^\pi(s,a)$$��

ͨ����˵����$$Q^w(s,a)$$���ƽ�$$Q^\pi(s,a)$$������ƫ�Ϊ������ƫ�Ӧ���㣺

1. $$Q^w(s,a)=\nabla_\theta\log\pi_\theta(a|s)^Tw$$������������Եĺ����ƽ�Ϊ���ԣ�
2. ����$$w$$��Ӧ����С���������$$\varepsilon^2(w)=\mathbb{E}_{\mathcal{S}\sim\rho^\pi,a\sim\pi_\theta}[(Q^w(s,a)-Q^\pi(s,a))^2]$$

��ô���ǵĲ����ݶ�Ϊ��

$$
\nabla_\theta J(\pi_\theta)=\mathbb{E}_{\mathcal{S}\sim\rho^\pi,a\sim\pi_\theta}[\nabla_\theta log\pi_\theta(a|s)Q^w(s,a)]
$$

��ʵ���У��ſ�����2�����������㷨ͨ��ʱ����ѧϰ������Ч�������������������1��2ȫ������Ļ��������㷨�൱�ڲ���Ҫʹ��critic��

### ����actor-criticѧϰ��Off-Policy Actor-Critic��*

�ӹ켣�в�ͬ�Ĳ�����Ϊ$$\beta(a|s) \neq \pi_\theta(a|s)$$�����߹��Ʋ����ݶ��Ǻܳ��õķ����������߻����£�����Ŀ��ͨ���޸�ΪĿ����Եļ�ֵ������

$$
\begin{aligned}
J_\beta(\pi_\theta)&=\int_\mathcal{S}\rho^{\beta}(s)V^\pi(s)ds \\
                   &=\int_\mathcal{S}\int_\mathcal{A}\rho^\beta(s)\pi_\theta(a|s)Q^\pi(s,a)dads
\end{aligned}
$$

�ݶ�ԼΪ

$$
\begin{aligned}
\nabla_\theta J_\beta(\pi_\theta) &\approx \int_\mathcal{S}\int_\mathcal{A}\rho^\beta(s)\nabla_\theta\pi_\theta(a|s)Q^\pi(s,a)dads \\
                                  &= \mathbb{E}_{\mathcal{S}\sim\rho^\beta,a\sim\beta}[\frac{\pi_\theta(a|s)}{\beta_\theta(a|s)}\nabla_\theta \log \pi_\theta(a|s)Q^\pi(s,a)]
\end{aligned}
$$

����actor-critic�㷨��OffPAC��ʹ����Ϊ���ԣ�behaviour  policy��$$\beta(a|s)$$�����ɹ켣��critic��һ��״̬-��ֵ�ĺ���$$V^v(s)\approx V^\pi(s)$$��actor�������²��ԵĲ���$$\theta$$��actor��critc����ͨ�����ߵĹ켣���ݽ���ѵ�����͹�ʽ��$$Q^\pi(s,a)$$��ͬ���ǣ�����ʹ��ʱ�������$$\delta_t=r_{t+1}+\gamma V^v(s_{t+1})-V^v(s_t)$$�����������ṩ��ʵ�ݶȵĽ���ֵ������ʹ��$$\frac{\pi_\theta(a|s)}{\beta_\theta(a|s)}$$��һ�����ж�action�����Ǹ��ݲ���$$\pi$$����$$\beta$$��

## 

## ȷ���Բ����ݶȣ�Gradients of Deterministic Policies��

���ڿ�����ν������ݶȿ����չ��ȷ���Բ��ԡ�������ǰ�沿���������������ݶȶ���ȷ���Բ����ݶ�ʵ��������������ݶȵ�һ��������

### ����-ֵ�����ݶȣ�Action-Value Gradients��

�����model-freeǿ��ѧϰ���ǻ���һ��Ĳ��Ե��������������Ͳ��Ը��½������(Sutton and Barto, 1998)������������ͨ�����ؿ����ʱ�����㷨�������ݣ����$$Q^\pi(s,a)$$��$$Q^\mu(s,a)$$��������㷨�ǲ���̰���㷨����󻯶���-ֵ������$$\mu^{k+1}(s)=\underset{a}{\operatorname{argmax}} Q^{\mu^k}(s,a)$$��

�����ڶ����ռ�����������£�̰���㷨��Ҫ��ÿһ������󻯣��Ͳ��������⡣һ���򵥵���������ǽ�������$$Q$$���ݶȷ����ƶ���������ȫ�����$$Q$$��������˵������ÿһ��̽������״̬$$s$$����������Ĳ���$$\theta^{k+1}$$��$$\nabla_\theta Q^{\mu^k}(s, \mu_\theta(s))$$��һ�����������¡�ÿ����ͬ��״̬�����ṩ��һ�����µķ������з���ľ�ֵ�����Կ���$$\rho^\mu(s)$$��

$$
\theta^{k+1}:=\theta^k+\alpha\mathbb{E}_{\mathcal{S}\sim\rho^{\mu^k}}[\nabla_\theta Q^{\mu^k}(s,\mu_\theta(s))]
$$

���Ը��¿��Էֽ�Ϊ����-ֵ�������ݶȺ��������Ե��ݶȸ���

$$
\theta^{k+1}:=\theta^k+\alpha\mathbb{E}_{\mathcal{S}\sim\rho^{\mu^k}}[\nabla_\theta Q^{\mu^k}(s,a)|_{a=\mu_\theta(s)}]
$$

���չ�����$$\nabla_\theta\mu_\theta(s)$$��һ���ſɱȾ���Ҳ����˵��ÿһ�ж����ݶ�$$\nabla_\theta[\mu_\theta(s)]_d$$��$$d$$�Ƕ����ռ��ά�ȣ���ͨ���ı���ԣ���ͬ��״̬���ᱻ̽��������״̬�ֲ�$$\rho^\mu$$Ҳ�ᱻ�ı䡣

### ȷ���Բ����ݶȶ���Deterministic Policy Gradient Theorem��

���ڿ��Ǵ��в�������$$\theta\in\mathbb{R}^n$$ȷ���Բ���$$\mu_\theta:\mathcal{S}\to\mathcal{A}$$������Ŀ�꺯��Ϊ$$J(\mu_\theta)=\mathbb{E}[r^\gamma_1|\mu]$$��������ʷֲ�$$p(s\to s',t,\mu)$$���ۿ�״̬�ֲ�$$\rho^\mu(s)$$����Ŀ�꺯��дΪ������

$$
\begin{aligned}
J(\mu_\theta)&=\int_{\mathcal{S}}\rho^\mu(s)r(s,\mu_\theta(s))ds \\
             &=\mathbb{E}_{\mathcal{S}\sim\rho^\mu}[r(s,\mu_\theta(s))]
\end{aligned}
$$


#### ����1(Deterministic  Policy  Gradient  Theorem)

���MDP��������$$p(s'|s, a)$$��$$\nabla_a p(s'|s, a)$$��$$\mu_\theta(s)$$��$$\nabla_\theta\mu_\theta(s)$$��$$r(s,a)$$��$$\nabla_a r(s,a)$$��$$p_1(s)$$���ڲ���$$s$$, $$a$$, $$s'$$, $$x$$�¶��������ģ���ô��ζ��$$\nabla_\theta\mu_\theta(s)$$��$$\nabla_aQ^\mu(s,a)$$������ȷ���Բ����ݶȴ��ڡ���ô��

$$
\begin{aligned}
\nabla_\theta J(\mu_\theta)&=\int_{\mathcal{S}}\rho^\mu(s)\nabla_\theta\mu_\theta(s)\nabla_aQ^\mu(s,a)|_{a=\mu_\theta(s)}ds \\
             &=\mathbb{E}_{\mathcal{S}\sim\rho^\mu}[\nabla_\theta\mu_\theta(s)\nabla_aQ^\mu(s,a)|_{a=\mu_\theta(s)}]
\end{aligned}
$$


##### ֤��

���MDP����$$p(s'|s, a)$$��$$\nabla_a p(s'|s, a)$$��$$\mu_\theta(s)$$��$$\nabla_\theta\mu_\theta(s)$$��$$r(s,a)$$��$$\nabla_a r(s,a)$$��$$p_1(s)$$���ڲ���$$s$$, $$a$$, $$s'$$, $$x$$�¶��������ģ���ô$$V^{\mu_\theta}(s)$$��$$\nabla_\theta V^{\mu_\theta}(s)$$����$$\theta$$��$$s$$���������ġ���������$$\theta$$��״̬�ռ�$$\mathcal{S}$$�ǽ��յģ����$$||\nabla_\theta V^{\mu_\theta}{(s)}||$$��$$||\nabla_a Q^{\mu_\theta}(s,a)|_{a=\mu_\theta(s)}||$$��$$\nabla_\theta\mu_\theta(s)$$����$$s$$���н纯����

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

������һ������ȥ$$V^{\mu_\theta}$$:

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

�����Ƶ������У��м������õ����ݶȵ���ʽ����

$$
\begin{aligned}
\frac{df(a,g(x))}{dx}&=\frac{df(a,y)}{dx}|_{y=g(x)} \\
                     &=\frac{dy}{dx} \cdot \frac{df(a,y)}{dy}|_{y=g(x)} \\
                     &=\frac{dg(x)}{dx} \cdot \frac{df(a,y)}{dy}|_{y=g(x)}
\end{aligned}
$$

��ˣ������У�

$$
\begin{aligned}
\nabla_\theta J(\mu_\theta)&=\nabla_\theta\int_\mathcal{S}p_1(s)V^{\mu_\theta}(s)ds \\
    &=\int_\mathcal{S}p_1(s)\nabla_\theta V^{\mu_\theta}(s)ds \\
    &=\int_\mathcal{S}\int_\mathcal{S}\sum_{t=0}^\infty \gamma^t p_1(s)p(s\to s',t,\mu_\theta)\nabla_\theta\mu_\theta(s')\nabla_a Q^{\mu_\theta}(s',a)|_{a=\mu_\theta(s')}ds'ds \\
    &=\int_\mathcal{S} \rho^{\mu_\theta}(s)\nabla_\theta\mu_\theta(s)\nabla_a Q^{\mu_\theta}(s,a)|_{a=\mu_\theta(s)}ds
\end{aligned}
$$


#### ȷ���Բ�������������ݶȵ��������

���ǰ��������$$\pi_{\mu_\theta,\sigma}$$��ȷ������$$\mu_\theta:\mathcal{S}\to \mathcal{A}$$�ͷ���$$\sigma$$ �����棬��$$\sigma=0$$ʱ��$$\pi_{\mu_\theta,0}\equiv \mu_\theta$$��������Ե�ͬ��ȷ���Բ��ԡ�

$$
\lim_{\sigma \downarrow 0}\nabla_\theta J(\pi_{\mu_\theta})=\nabla_\theta J(\mu_\theta)
$$

(�¼�ͷ��ָ$$\sigma$$�����ݼ�������0)

##### ֤��*

// TODO

��http://proceedings.mlr.press/v32/silver14-supp.pdf��

### ȷ����actor-critic�㷨��Deterministic Actor-Critic Algorithms��

#### ����ȷ����actor-critic��On-Policy Deterministic Actor-Critic��

critic������������-ֵ������actor���ݶ���-ֵ���������ݶ�������actor�������¹�ʽ��������$$\theta$$

$$
\begin{aligned}
\nabla_\theta J(\mu_\theta)&=\int_{\mathcal{S}}\rho^\mu(s)\nabla_\theta\mu_\theta(s)\nabla_aQ^\mu(s,a)|_{a=\mu_\theta(s)}ds \\
             &=\mathbb{E}_{\mathcal{S}\sim\rho^\mu}[\nabla_\theta\mu_\theta(s)\nabla_aQ^\mu(s,a)|_{a=\mu_\theta(s)}]
\end{aligned}
$$

����������ݶ�һ���������ÿ�΢�ֵĶ���-ֵ����$$Q^w(s,a)$$��������ʵ�Ķ���-ֵ����$$Q^\mu(s,a)$$��critic�����þ��ǲ����ʵ��ĵ����㷨ʹ$$Q^w(s,a)\simeq Q^\mu(s,a)$$�����磬����Sarsa���µĻ���

$$
\begin{aligned}
\sigma_t &= r_t+\gamma Q^w(s_{t+1)},a_{(t+1)})-Q^w(s_t,a_t) \\
w_{t+1} &= w_t+\alpha_w\sigma_t\nabla_w Q^w(s_t,a_t) \\
\theta_{t+1} &=\theta_t+\alpha_\theta\nabla_\theta\mu_\theta(s_t)\nabla_a Q^w(s_t,a_t)|_{a=\mu_\theta(s)}
\end{aligned}
$$


#### ����ȷ���Բ��ԣ�Off-Policy Deterministic Actor-Critic��

���ڿ������߲��ԣ�$$\mu_\theta(s)$$��һϵ�й켣������ѧϰ�����ǵ�Ŀ�꺯��Ϊ��

$$
\begin{aligned}
J_\beta(\mu_\theta)&=\int_\mathcal{S}\rho^\beta(s)V^\mu(s)ds \\
  &= \int_\mathcal{S}\rho^\beta(s)Q^\mu(s,\mu_\theta(s))ds 
\end{aligned}
$$

�ݶ�Ϊ��

$$
\begin{aligned}
\nabla_\theta J\beta(\mu_\theta)&\simeq \int_\mathcal{S}\rho^\beta(s)\nabla_\theta\mu_\theta(a|s)Q^\mu(s,a)ds \\
  &= \mathbb{E}_{\mathcal{S}\sim \rho^\beta}[\nabla_\theta\mu_\theta(s)\nabla_aQ\mu(s,a)|_{a=\mu_\theta(s)}]
\end{aligned}
$$

ͬ���ģ��ÿ�΢�ֵĶ���-ֵ����$$Q^w(s,a)$$��������ʵ�Ķ���-ֵ����$$Q^\mu(s,a)$$������ѧϰͨ��$$\beta(a|s)$$�����켣���ݡ�������ciritcͨ��Q-learning��ѧϰ���Խ��и��£�

$$
\begin{aligned}
\sigma_t &= r_t+\gamma Q^w(s_{t+1},\mu_\theta(s_{t+1}))-Q^w(s_t,a_t) \\
w_{t+1}  &= w_t+\alpha_w\sigma_t\nabla_wQ^w(s_t,a_t) \\
\theta_{t+1}&=\theta_t+\alpha_\theta\nabla_\theta\mu_\theta(s_t)\nabla_aQ^w(s_t,a_t)|_{s=\mu_\theta(s)}
\end{aligned}
$$


#### Compatible Function Approximation

�ý��Ƶ�$$Q^w$$������ʵ�������һ����������ݶ�����ʵ������ݶȡ�

������ƺ���$$Q^w(s,a)$$��$$\mu_\theta(s)$$��$$\nabla_\theta J_\beta(\mu^\theta)=\mathbb{E}_{\mathcal{S}\sim\rho^\beta}[\nabla_\theta\mu_\theta(s)\nabla_aQ^\mu(s,a)|_{a=\mu_\theta}]$$���ݣ���ô������������������

- $$\nabla_aQ^w(s,a)|_{a=\mu_\theta(s)}=\nabla_\theta\mu_\theta(s)^Tw$$
- $$w$$yӦ��С���������$$\mathrm{MSE}(\theta,w)=\mathbb{E}[\epsilon(s;\theta,w)^T\epsilon(s;\theta,w)]$$�����У�$$\epsilon(s;\theta,w)=\nabla_aQ^w(s,a)|_{a=\mu_\theta(s)}-\nabla_aQ^\mu(s,a)|_{a=\mu_\theta(s)}$$

##### ֤��*

���$$w$$��С�� MSE����ô$$\epsilon^2$$��$$w$$ ���ݶ�һ����0����������1��$$\nabla_w\epsilon(s;\theta,w)=\nabla_\theta\mu_\theta(s)$$��

$$
\nabla_w\mathrm{MSE}(\theta,w)=0 \\
\mathbb{E}[\nabla_\theta\mu_\theta(s)\epsilon(s;\theta,w)]=0 \\
\mathbb{E}[\nabla_\theta\mu_\theta(s)\nabla_aQ^w(s,a)|_{a=\mu_\theta(s)}]=\mathbb{E}[\nabla_\theta\mu_\theta(s)\nabla_aQ^\mu(s,a)|_{a=\mu_\theta(s)}] \\
=\nabla_\theta J_\beta(\mu_\theta) \ \mathrm{or}\ \nabla_\theta J(\mu_\theta)
$$

//TODO: