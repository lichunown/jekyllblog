---
layout:     post
title:      "tencorflow在windows下的安装"
subtitle:   "折腾，捣鼓。。。"
date:       2017-02-21
author:     "LCY"
header-img: "img/in-post/install-tensorflow/index.png"
tags:
    - python
    - tensorflow
---

最近想看一下tencorflow, 然而这个库的限制太多了。windows版本的只支持3.5版本。本来想用docker，继续使用python2.7来。但是，坑爹的docker配了我好几天始终配不好。
只好换用python3.5了。

### 安装python3.5

<br>
Anaconda是个好东西，一键式管理库。我安装的是Anaconda2，对应于python2.7版本。
[Anaconda下载](https://www.continuum.io/downloads)
<br>
然而这次要用python3.5版本，怎么办嘞。。。
<br><br><br>

**安装**

```bash
conda create -n py35 python=3.5 anaconda
```
嗯，一行命令自动配置

<br><br><br>
**切换版本时**
```bash
activate py35
```
当然，很有可能因为网络问题安装失败。<br>
这时，如果要重新来过的话。

```bash
activate py35
conda install anaconda  python=3.5
```
----------------
**关于网络。。。**
shadowsocks是个好东西，然而在命令行下就很难用了。<br>
有个软件叫*SocksCap64*，用它启动cmd可以使用shadowsocks的代理。
----------------
### 安装tensorflow

官方网站[Installing TensorFlow on Windows](https://www.tensorflow.org/install/install_windows)提供了两种安装方法。我使用的第一种，直接pip安装。

```bash
pip3 install --upgrade tensorflow
```

### 检查安装情况

进入python
```bash
python
```
测试
```python
>>> import tensorflow as tf




hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

```
如果输出如下：
```python
Hello, TensorFlow!
```

就应该没问题了。

*然而我测试的在正常输出前有一段报错，并不知道原因，不过问题应该不大*

###最后的最后

开始学习tensorflow吧。[tensorflow文档加部分翻译](/data/tensorflow-doc.pdf)