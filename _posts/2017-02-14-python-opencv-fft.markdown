---
layout:     post
title:      "python下使用opencv库实现图片滤波"
subtitle:   "总结整理"
date:       2017-02-14
author:     "LCY"
header-img: "img/in-post/python-opencv/fft.jpg"
tags:
    - python
    - opencv
---



嗯，整理一下当时搞opencv时候的一些心得。

### opencv的安装

opencv是一个很有名气的开源图像处理的库。支持多种语言。<br>
它的下载地址：[opencv下载](http://opencv.org/downloads.html)<br>

下载下来后是一个压缩包，解压，找到目录` opencv/build/python/2.7/x64/cv2.pyd`的这个cv2.pyd文件。当然，我的python版本是64位的2.7版本。**一定要对应好python版本**。<br>

然后，将这个文件复制到python安装目录里的`./Lib/site-packages/`。<br>

最后，运行python，执行

```python
import cv2
```

检查一下能否加载成功。

### 关于图片的傅里叶变换

嗯，我还没有学信号与系统课程。所以对这块理解并不是很深入。不过我当时学习时找到了一份很全面的讲解。<br>

[傅里叶分析之掐死教程](https://zhuanlan.zhihu.com/p/19763358)

而图像的傅里叶变换，只是由一维变成了二维，将图片从时域变换到频域。通过傅里叶变换，可以简单的分离出图像的高频部分与低频部分。而当对高频与低频部分分别进行一些处理，再进行傅里叶逆变换后，就会时图片呈现出很多特殊的效果。简单的说，高频部分对应图像的轮廓，而低频部分对应图像的细节。<br>

在这里盗两张图：

![去除高频部分](/img/in-post/python-opencv/lowpass.jpg)

*注：去除高频部分（高通滤波器）*

![去除低频部分](/img/in-post/python-opencv/highpass.jpg)

*注：去除低频部分（低通滤波器）*

图片来源以及更详细的讲解：

[Python下opencv使用笔记（十）（图像频域滤波与傅里叶变换）](http://blog.csdn.net/on2way/article/details/46981825)



### 开始搞事情

嗯，我干的具体事情，明天再写。。。
<br>
**mark一下**