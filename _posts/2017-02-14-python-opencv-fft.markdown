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
### 前期准备

#### opencv的安装

opencv是一个很有名气的开源图像处理的库。支持多种语言。<br>
它的下载地址：[opencv下载](http://opencv.org/downloads.html)<br>

下载下来后是一个压缩包，解压，找到目录` opencv/build/python/2.7/x64/cv2.pyd`的这个cv2.pyd文件。当然，我的python版本是64位的2.7版本。**一定要对应好python版本**。<br>

然后，将这个文件复制到python安装目录里的`./Lib/site-packages/`。<br>

最后，运行python，执行

```python
import cv2
```

检查一下能否加载成功。

#### numpy，matplotlib等数学工具包

opencv读取图像后，以矩阵的形式存储，展示。因此，我们需要使用到python处理矩阵的`numpy`。而绘图，我用到的是`matplotlib`包。<br>
安装这两个包，就十分简单了。
```bash
pip install numpy
pip install matplotlib
```

### 关于图片的傅里叶变换

嗯，我还没有学信号与系统课程。所以对这块理解并不是很深入。不过我当时学习时找到了一份很全面的讲解。<br>

[傅里叶分析之掐死教程](https://zhuanlan.zhihu.com/p/19763358)

而图像的傅里叶变换，只是由一维变成了二维，将图片从时域变换到频域。通过傅里叶变换，可以简单的分离出图像的高频部分与低频部分。而当对高频与低频部分分别进行一些处理，再进行傅里叶逆变换后，就会时图片呈现出很多特殊的效果。简单的说，高频部分对应图像的轮廓，而低频部分对应图像的细节。<br>

在这里盗两张图：

![去除高频部分](/img/in-post/python-opencv/lowpass.jpg)

*注：去除高频部分（低通滤波器）*

![去除低频部分](/img/in-post/python-opencv/highpass.jpg)

*注：去除低频部分（高通滤波器）*

图片来源以及更详细的讲解：

[Python下opencv使用笔记（十）（图像频域滤波与傅里叶变换）](http://blog.csdn.net/on2way/article/details/46981825)



### 开始搞事情

#### opencv的基本用法

##### 读取图片

```python
img = cv2.imread('test.jpg')
```

这种情况下，默认读取的是三原色的RGB图。即使原图为灰色，读取出来的也是有3个通道。这种情况下，图片是一个三维的矩阵。`img[:,:,0]`是蓝色的值（B），`img[:,:,1]`是绿色的值（G），`img[:,:,2]`是红色的值（R）。**注意，顺序刚好和RGB相反**。

而很多情况下，我们为了简化操作，更希望读入的图片只是一个二维的矩阵，也就是读入一个灰色的图片。

```python
img = cv2.imread('test2.jpg',cv2.IMREAD_GRAYSCALE)
```
参数的解释：

| 色彩空间 |         常量名          |  值   |
| :--: | :------------------: | :--: |
|  灰色  | cv2.IMREAD_GRAYSCALE |  0   |
|  彩色  |   cv2.IMREAD_COLOR   |  1   |



当然，也可以先读取一个图片，然后再对它进行色彩空间的转换。

```python
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # BGR 转换到 灰色
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)    # BGR 转换到 HSV [Hue(色调),Saturation(饱和度)和Value(亮度)]  
```

多说一句HSV的颜色空间。

> RGB是为了让机器更好的显示图像,对于人类来说并不直观,HSV更为贴近我们的认知,所以通常我们在针对**某种颜色做提取**时会转换到HSV颜色空间里面来处理. 
>
> 需要注意的是H的取值范围为0°～360°，从红色开始按逆时针方向计算，红色为0°，绿色为120°,蓝色为240°，想提取红色时需注意取值应为-10~10（打比方）**OpenCV中H的取值范围为0~180(8bit存储时)**,

![HSV色彩空间](/img/in-post/python-opencv/HSV.jpg)

##### 保存图片

```python
cv2.imwrite(name,img)
```




#### numpy的傅里叶变换函数

傅里叶变换是一种数学方法，嗯……所以numpy就足以兼任这一图片处理的重任了（比opencv库里的直观，还简单）。

```python
fft = numpy.fft.fft2(img)              # 傅里叶变换
fftshift = numpy.fft.fftshift(fft)     # 中心化操作
fft = numpy.fft.ifftshift(fftshift)    # 逆中心化操作
img = numpy.fft.ifft2(fft)             # 逆傅里叶变换
```

> 程序中为什么要有一个np.fft.fftshift(f)中心化操作？整个图像是在傅里叶变换的一个周期内完成的，将其看成横纵两个方向的一维傅里叶变换，在每个方向上都会有高频信号和低频信号，那么傅里叶变换将低频信号放在了边缘，高频信号放在了中间，然而一副图像，很明显的低频信号多而明显，所以将低频信号采用一种方法移到中间，在时域上就是对f乘以（-1）^(M+N)，换到频域里面就是位置的移到了。

在这里，中心化图片要执行两次函数，略微感到麻烦。因此，我直接将之打包一下

```python
def fft(img):
    return numpy.fft.fft2(img)
def fftshift(img):
    return numpy.fft.fftshift(fft(img))
def ifft(img):
    return numpy.fft.ifft2(img)
def ifftshift(img):
    return ifft(numpy.fft.ifftshift(img))
```

然后，傅里叶加中心化，只需`fftshift(img)`，倒回去呢，只需`ifftshift(img_fftshift)`。至少看起来简洁不少。

#### 滤波

前文提到过，中心化之后，中心是高频信号，周围是低频信号。那么，对于这些波，你想怎么处理就怎么处理喽。

我介绍一种最简单的滤波方法。（嗯，又是盗的）

创建一个与原图大小相同的矩阵，值分别是0，1。这样，使两个矩阵内值对应相乘，就可以滤掉矩阵值是0的位置的数据。我们可以创建一个中心是0，周围是1的矩阵，相乘即实现了低通滤波。而中心是1，周围是0，就实现了高通滤波。

```python
def mask(img,rate = 0.1):  # 产生一个滤波矩阵
    mask = numpy.ones(img.shape) # 创建全是1的与原图大小相同的矩阵
    rows,cols = img.shape# 图片的长宽
    mask[rows/2-rows*rate:rows/2+rows*rate,cols/2-cols*rate:cols/2+cols*rate] = 0 # 使中心方块区域成为0
    # 区域大小取决于rate
    # plt.imshow(mask)
    return mask
```

滤波

```python
img = cv2.imread('test.img',0)              # 读取图片 
fftshift_img = fftshift(img)                # 傅里叶变换加中心化
fftshift_result = fftshift_img * mask(img)  # 滤波
result = ifftshift(fftshift_result)         # 逆变化，查看原图
```
这是原图：
![原图](/img/in-post/python-opencv/encode.jpg)
高通滤波之后：
![滤波](/img/in-post/python-opencv/result1.jpg)