---
layout:     post
title:      "python高级用法及编码规范"
subtitle:   "开发过程常见问题笔记"
date:       2018-12-29
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

## abc(Abstract Base Classes)

声明抽象函数

```python
import abc
 
class Animal(metaclass=abc.ABCMeta):
 
    @abc.abstractmethod
    def screaming(self):
        'Return when animal screaming the sound hear likes'
        return NotImplemented
 
    @abc.abstractmethod
    def walk(self, x, y):
        'Make animal walk to position (x, y).'
        return NotImplemented
```

```python
>>> class Dog(Animal):
...    pass
...
>>> Dog()  # Create a instance
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: Can't instantiate abstract class Dog with abstract methods screaming, walk
```

支持classmethod，property

```python
import abc
 
class Base(abc.ABC):
 
    @classmethod
    @abc.abstractmethod
    def setUpClass(cls):
        return NotImplemented
 
    @staticmethod
    @abc.abstractmethod
    def count(self, data):
        return len(data)
 
 
class Implementation(Base):
 
    @classmethod
    def setUpClass(cls):
        cls.count = 0
 
    @staticmethod
    def count(self, data):
        self.count = len(data)
        return self.count
    
    
import abc
 
 
class Base(abc.ABC):
    _index = 0
 
    @property
    @abc.abstractmethod
    def index(self):
        return self._index
 
    @index.setter
    @abc.abstractmethod
    def index(self, new_index):
        self._index = new_index
 
 
class Implementation(Base):
    MAX_LEN = 100
 
    @property
    def index(self):
        return self._index
 
    @index.setter
    def index(self, new_index):
        new_index = min(new_index, self.MAX_LEN)
        self._index = new_index
 
imp = Implementation()
print(imp.index)
imp.index = 50
print(imp.index)
imp.index = 500
print(imp.index)
```

## weakref 弱引用

可以处理循环引用问题

```python
import sys  # We can use sys.getrefcount(obj) to get refcnt
 
 
class Foo(object):
    pass
 
 
# Create Foo Object A
# And print reference count
A = Foo()
print('Refcnt of A: ', sys.getrefcount(A))
 
# Create a strong reference to A
# And check if B is reference to A
# Then print A refcnt, it should +1
B = A
print("A is B's referent: ", id(B) == id(A))
print('Refcnt of A: ', sys.getrefcount(A))
 
# Create three strong reference to A
# Refcnt of A should +3
C = A
D = A
E = A
print('Refcnt of A: ', sys.getrefcount(A))
 
# Delete E should -1 at A's refcnt
del E
print('Refcnt of A: ', sys.getrefcount(A))
```

```python
import sys
import weakref
 
 
class Foo(object):
    def show(self):
        print('hello')
 
 
A = Foo()
print('Refcnt of A: ', sys.getrefcount(A))
 
B = weakref.ref(A)
print(B)
B().show()
print('Refcnt of A: ', sys.getrefcount(A))
```



