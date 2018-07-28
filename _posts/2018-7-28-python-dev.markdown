---
layout:     post
title:      "python高级用法及编码规范"
subtitle:   "开发过程常见问题笔记"
date:       2018-7-28
author:     "LCY"
header-img: "img/default.jpg"
tags:
    - python
---

## 反射

反射是在只知道类名或者函数名的情况下调用其对应的函数

对于本文件内的全局变量,可以使用`globals()`函数获取全局变量 名称-值 的键值对

```python
# return a dict    name -> value
glodict = globals()


def foo():
    dosomething()
    pass

# run `foo()`
globals()['foo']()
```

对于类内变量, 使用`__dict__`

```python
class Foo(object):
    def __init__(self, **kwargs):
        self.value1 = 1
        self.value2 = 2
        
        for name in kwargs:
            if name in self.__dict__:
                self.__dict__[name] = kwargs[name]
                
    def dosomething(self):
        dosomething_()

t = Foo(value1 = 100)
# t.value1 == 100 & t.value2 == 2
```

路由  eg:

```python
class Dog(object):

    def __int__(self, name):
        self.name = name

    def eat(self, food):
        print('I love eat '+food)

def rout(url):
    urls = url.split("&")
    map = {}
    for uri in urls:
        u = uri.split('=')
        map[u[0]] = u[1]
    obj = globals()[map.get('cls')]()
    func = getattr(obj, map.get('method'))
    func(map.get('value'))#I love eat meat

rout("cls=Dog&method=eat&value=meat")

```
## 动态继承

eg:

对于图片来说,可能有PNG, JPG等格式

如果我们有一个ImageZip类, 无法确定继承自那种Image格式(PNGImage or JPGImage ? )

使用工厂函数自动创建类:

```python
def image_factory(path):
    # ...
    if format == ".gz":
        image = unpack_gz(path)
        format = os.path.splitext(image)[1][1:]
        if format == "jpg":
            return MakeImageZip(ImageJPG, image)
        elif format == "png":
            return MakeImageZip(ImagePNG, image)
        else: raise Exception('The format "' + format + '" is not supported.')

def MakeImageZIP(base, path):
    '''`base` either ImageJPG or ImagePNG.'''
    class ImageZIP(base):
        # ...

    return  ImageZIP(path)

def ImageZIP(path):

    path = unpack_gz(path)
    format = os.path.splitext(image)[1][1:]

    if format == "jpg": base = ImageJPG
    elif format == "png": base = ImagePNG
    else: raise_unsupported_format_error()

    class ImageZIP(base): # would it be better to use   ImageZip_.__name__ = "ImageZIP" ?
        # ...

    return ImageZIP(path)
```

## hack 输出流

当项目有很多文件时，要找出控制台的输出是在哪里print出来的很麻烦 

```python

import sys,traceback
class mystdout:
    stdout = sys.stdout
    def write(self,_str):
        if _str != '\n':
            filepath,lineno = traceback.extract_stack()[-2][0:2]
            mystdout.stdout.write("%s\t%s(%s)\n"%(_str,filepath,lineno))
 
sys.stdout = mystdout()
 
print 'foo'
print 'bar'

# 输出
# foo test_stdout.py(11)
# bar test_stdout.py(12)
```

当`print 'foo'`的时候，会调用`sys.stdout.write()`，不过因为`sys.stdout = mystdout()`，被重写了，所以实际调用的是`mystdout`类的`write()`方法。 在python中`print`会自动加换行符'\n',而且是单独`sys.stdout.write('\n')`,所以要`if _str != '\n'`。 再加上`traceback`获得文件名和行号，这样控制台的每个输出都能快速定位到在哪里print的了。 