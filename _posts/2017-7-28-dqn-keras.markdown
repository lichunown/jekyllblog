---
layout:     post
title:      "基于keras的增强学习实战"
subtitle:   "用DQN来玩游戏"
date:       2017-07-28
author:     "LCY"
header-img: "img/default.jpg"
tags:
    - ML
    - python
---

## 什么是DQN
深度增强学习(Deep Reinforcement Learning)是将深度学习与增强学习结合起来从而实现从Perception感知到Action动作的端对端学习End-to-End Learning的一种全新的算法。简单的说，就是和人类一样，输入感知信息比如视觉，然后通过深度神经网络，直接输出动作，中间没有hand-crafted engineering的工作。深度增强学习具备使机器人实现真正完全自主的学习一种甚至多种技能的潜力。

关于DQN的原理，大佬们已经有一个系列的blog文章。

[DQN 从入门到放弃1 DQN与增强学习](http://blog.csdn.net/AMDS123/article/details/70242419?locationNum=13&fps=1)
[DQN 从入门到放弃2 增强学习与MDP](https://zhuanlan.zhihu.com/p/21292697?refer=intelligentunit)
[DQN 从入门到放弃3 价值函数与Bellman方程](https://zhuanlan.zhihu.com/p/21340755?refer=intelligentunit)
[DQN 从入门到放弃4 动态规划与Q-Learning](https://zhuanlan.zhihu.com/p/21378532?refer=intelligentunit)
[DQN从入门到放弃5 深度解读DQN算法](https://zhuanlan.zhihu.com/p/21421729)

<-----------------------------以下扩充学习-------------------------------------->

[DQN从入门到放弃6 DQN的各种改进](https://zhuanlan.zhihu.com/p/21547911?refer=intelligentunit)
[ DQN从入门到放弃7 连续控制DQN算法-NAF ](http://c.colabug.com/thread-1597875-1-1.html)



## 实现一个简单的DQN demo

理论如果看不下去，那我们直接看代码吧。。。

### 游戏准备

我要安利一个python的库gym。[OpenAI Gym](https://gym.openai.com/)

gym集成了很多小游戏，专门用来进行机器学习这一方面的处理。

按照官网上的介绍，我们可以很方便的实现一个`CartPole-v1`游戏。

```python
import gym,random

env = gym.make('CartPole-v1')
health = 3
for e in range(health):# 游戏局数
    state = env.reset()
    while True:# 每局游戏中的循环
        action = random.sample([0,1],1)[0]#随机产生动作
        next_state, reward, done, _ = env.step(action)
        if done:# 游戏失败退出循环
            print("[Fail] health:{} times:{}".format(e))
            break
```

在这里，实际上主要就用了两条命令

- `state = env.reset()`: 初始化游戏环境。

- `next_state, reward, done, _ = env.step(action)`: 对游戏中的每一个动作产生反应。

  在'CartPole-v1'游戏中，action为一个0或者1的值，分别对应于方块的移动。这一函数返回4个值，分别是：

  - next_state：下一个游戏状态
  - reward：奖赏（在这个游戏中恒为1，对我们实际上没有用）
  - done：游戏是否结束
  - _： 其他信息（这个游戏也是空的）


游戏过程搞懂了，接下来就是实现我们的DQN算法了。

### agent设计

首先，设计我们的类：

```python
class DQN(object):
    def __init__(self,state_size, action_size):# state_size:输入矩阵大小，action_size输出矩阵大小
        pass
    def _createModel(self):# 建立神经网络模型
        pass
    def remember(self,state,action,reward,next_state,done):# 对过程进行记忆
        pass
    def train(self):# 训练神经网络
        pass
    def predict_action(self,action):# 调用神经网络进行最优预测
        pass
    def act(self,action):# 训练过程中执行的下一步动作（根据epsilon决定随机还是预测）
        pass
    
```

### 主函数实现

先不管具体的实现，我们先改写我们的主函数：

```python
import gym
import random

env = gym.make('CartPole-v1')
EPISODES = 3000

state_size = env.observation_space.shape[0] # 输入矩阵大小 4
action_size = env.action_space.n #输出矩阵大小(one hot) 2
agent = DQN(state_size,action_size)  

for e in range(EPISODES):# 游戏局数
    state = env.reset()
    while True:# 每局游戏中的循环
        action = agent.act(state)# 预测动作
        next_state, reward, done, _ = env.step(action)
        reward = 0.1 if not done else -1 # 更改奖赏值
        next_state = next_state.reshape(1,state_size)
        agent.remember(state,action,reward,next_state,done)# 记忆
        state = next_state
        if done:# 游戏失败退出循环
            print("[Fail] health:{} times:{}".format(e))
            break
    agent.train()# 训练数据
```



### DQN类内部过程的实现

看起来很完美，万事俱备，只差东风了。

####  \_\_init\_\_

```python
    def __init__(self,state_size, action_size,dd=True):
        self.state_size = state_size # 输入大小
        self.action_size = action_size # 输出大小
        self.memory = deque(maxlen=3000) # 初始化记忆
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.train_batch = 32
        self.model = self._createModel(dd) # 创建模型
```



**关于输入输出：**

相信学过机器学习的同学对这块应该不陌生，train_batch是只模型训练过程中一次训练batch的数据。输出为了方便期间，进行了onehot处理，不在是一个固定的0-1之间的值，而是一个2维的矩阵。

**关于epsilon**：

增强学习过程epsilon将会逐渐减少，`self.learning_rate`为最低的epsilon，`self.epsilon_decay`为削减率。



#### _createModel

```python
    def _createModel(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return model
```

使用keras设计神经网络，真的是很简洁，很爽。。。

关于keras，请查阅：[Keras中文文档](http://keras-cn.readthedocs.io/en/latest/)

#### train

```python
        if len(self.memory)>=self.train_batch: # 满足batch大小才可以进行训练
            minibatch = random.sample(self.memory,self.train_batch) # 随机从记忆中读数据
            state_batch = np.zeros([self.train_batch,self.state_size])#初始化batch数据
            target_batch = np.zeros([self.train_batch,self.action_size]) #初始化batch数据
            for i,(state, action, reward, next_state, done) in enumerate(minibatch): #初始化batch数据
                state_batch[i,:] = state
                target_batch[i,:] = self.predict_action(state)
                target_batch[i,action] = reward if done else reward+self.gamma*np.amax(self.predict_action(next_state)[0])
            self.model.fit(state_batch, target_batch, epochs=1, verbose=0)# 训练
            if self.epsilon > self.epsilon_min:# 更改epsilon
                self.epsilon *= self.epsilon_decay
```

关于`target_batch`，这就是DQN的精髓了。

由于我们是onehot，所以实际上只用预测一个值就好，另一个值，用原有的就可以。所以我们第一步先初始化`target_batch[i,:] = self.predict_action(state)`。

**然后就是用DQN的算法了** :  reward + $$\gamma$$ *predict_reward(next_state) 

关于这个算法，还是参考上面给出的连接[DQN 从入门到放弃4 动态规划与Q-Learning](https://zhuanlan.zhihu.com/p/21378532?refer=intelligentunit)

#### remember

```python
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
```

很简单，就是append。

值得一提的是，`self.memory`的初始化

```python
self.memory = deque(maxlen=3000) 
```

`deque`是python中的队列数据类型，可以赋一个最大存储数量。当超出这一数量，将会自动出队。

#### act & predict_action

```python
    def predict_action(self,state):# 预测动作
        return self.model.predict(state)
    def act(self,state):# 执行的动作，具有随机性
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.predict_action(state)[0])
```

### 然后。。。

就没了，就这么简单。

 **开不开心，惊不惊喜，意不意外。**










## 最后的最后

*源码大放送*

### 环境准备

必备库：

- numpy
- gym
- tensorflow
- keras

**参考**安装方式

```bash
pip install numpy
pip install gym
pip install tensorflow
pip install keras
```

### **源码**

```python
# -*- coding: utf-8 -*-

import gym
#from DQN_cart import DQN
import time
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os
from keras.models import load_model
from keras import backend as K
EPISODES = 3000

class DQN(object):
    def __init__(self,state_size, action_size,dd=True):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=3000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.train_batch = 32
        self._model = self._createModel(dd)
        
    @property
    def model(self):# 定义为只读属性
        return self._model
    
    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)
    
    def _createModel(self,dd=True):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        if dd:
            model.compile(loss=self._huber_loss,optimizer=Adam(lr=self.learning_rate))
        else:
            model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return model
        
    def train(self):
        if len(self.memory)>=self.train_batch:
            minibatch = random.sample(self.memory,self.train_batch) 
            state_batch = np.zeros([self.train_batch,self.state_size])
            target_batch = np.zeros([self.train_batch,self.action_size]) 
            for i,(state, action, reward, next_state, done) in enumerate(minibatch):
                state_batch[i,:] = state
                target_batch[i,:] = self.predict_action(state)
                target_batch[i,action] = reward if done else reward+self.gamma*np.amax(self.predict_action(next_state)[0])
            self.model.fit(state_batch, target_batch, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
    def predict_action(self,state):# 预测动作
        return self.model.predict(state)
    def act(self,state):# 执行的动作，具有随机性
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            #print(self.predict_action(state))
            return np.argmax(self.predict_action(state)[0])
        
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
        #self._train()
    def save(self,name = 'models/test'):
        self.model.save(name)
        self.saveWeight(name)
    def load(self,name = 'models/test'):
        self._model= load_model(name)
    def saveWeight(self,name = 'models/test'):
        self.model.save_weights(name+'.weight')
    def loadWeight(self,name = 'models/test'):
        self.model.load_weights(name+'.weight')
        
        
if __name__=='__main__':
    saveModelName = 'models/test'
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQN(state_size,action_size,False)   
    for times in range(EPISODES):
        state = env.reset().reshape(1,state_size) 
        for i in range(199):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = 0.1 if not done else -1
            next_state = next_state.reshape(1,state_size)
            agent.remember(state,action,reward,next_state,done)
            state = next_state
            if done:
                print('[times]:{}/{}\t\t[i]:{}\t\t[epsilon]:{}'.format(times,EPISODES,i,agent.epsilon))
                break
            if i == 198:
                print('[times]:{}/{}\t\t[i]:{}\t\t[epsilon]:{}\t#success#'.format(times,EPISODES,i,agent.epsilon))
        agent.train()
        if (times+1)%100==0:
            agent.save(saveModelName+str(times+1))
            print('[Saved] savename: `%s`'%(saveModelName+str(times+1)))
```

