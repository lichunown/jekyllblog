---
layout:     post
title:      "时间序列法的笔记"
subtitle:   "数模预测的好算法"
date:       2017-02-09
author:     "LCY"
header-img: "img/default.jpg"
tags:
    - matlab
    - 数模
---

### 滑动平均模型

$$

\frac{y=y_t+y_{t-1}+...+y_{t-N+1}}{N}

$$

### 加权滑动平均模型

$$

\frac{y=a_0y_t+a_1y_{t-1}+...+a_{N-1}y_{t-N+1}}{N}

$$

### 二次滑动平均

对经过一次滑动产生的序列再滑动平均

## 趋势向提取

```matlab
t = 1:0.1:10;
x = 10+2*t+sin(t);
y = detrend(x);%消除线性趋
y = detrend(x,'constant');
y = detrend(x,'linear',[10,12,13]);
plot(y);
plot(x-y);%提取线性趋势
```

## 插值

```matlab
y = interp(x,r)
[y,b] = interp(x,r,l,alpha)
%%
%  x:时间序列   
%  r:插入点倍数
%  l:c
%  alpha
%%
x = sin(t);
plot(x);hold on;stem(x);
y = interp(x,4);
plot(y);hold on;stem(y);
%% or
[y,b] = resample(x,p,q,n,beta)
```

### 离散点

```
x=[...];
y=[...];
yy = spline(x,y)%三次样条插值函数
xx = 1:0.1:10;
Y = ppval(yy,xx);
plot(x,y,'.',xx,Y);


```
