---
layout:     post
title:      "python的深复制与浅复制"
subtitle:   "这是个大坑（同样是以前的笔记）"
date:       2017-02-09
author:     "LCY"
header-img: "img/default.jpg"
tags:
    - python
---

#### 被坑了，调试了3天，才发现原因出在类的初始化上。

## 错误做法：

```python
class A(object):
    def __init__(self,list=[]):
        self._list=list
    def print(self):
        for i in self._list:
            print '%s,' % i
```
调用时

```python
a=A()
b=A()
a._list.append(1)
b._list.append(2)
id(a._list)
id(b._list)
```
结果，id出来的数相同……

## 正确做法：

> Python中的对象之间赋值时是按引用传递的，如果需要拷贝对象，需要使用标准库中的copy模块。 
> 1. copy.copy 浅拷贝 只拷贝父对象，不会拷贝对象的内部的子对象。 
> 2. copy.deepcopy 深拷贝 拷贝对象及其子对象 

```python
import copy  
a = [1, 2, 3, 4, ['a', 'b']] #原始对象  
  
b = a #赋值，传对象的引用  
c = copy.copy(a) #对象拷贝，浅拷贝  
d = copy.deepcopy(a) #对象拷贝，深拷贝  
  
a.append(5) #修改对象a  
a[4].append('c') #修改对象a中的['a', 'b']数组对象  
  
print 'a = ', a  
print 'b = ', b  
print 'c = ', c  
print 'd = ', d  
```

 输出结果： 

```python
a = [1, 2, 3, 4, ['a', 'b', 'c'], 5] 

b = [1, 2, 3, 4, ['a', 'b', 'c'], 5] 

c = [1, 2, 3, 4, ['a', 'b', 'c']] 

d = [1, 2, 3, 4, ['a', 'b']]
```