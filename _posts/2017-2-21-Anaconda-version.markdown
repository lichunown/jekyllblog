---
layout:     post
title:      "Anaconda管理python版本"
subtitle:   "一条命令协调一切"
date:       2017-02-21
author:     "LCY"
header-img: "img/default.jpg"
tags:
    - python
---

最近想看一下tencorflow, 然而这个库的限制太多了。windows版本的只支持3.5版本。本来想用docker，继续使用python2.7来。但是，坑爹的docker配了我好几天始终配不好。
只好换用python3.5了。
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
activate py34
```