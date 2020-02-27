---
layout:     post
title:      "gpulimit----开源项目安利"
subtitle:   "数学真有意思"
date:       2020-2-27
author:     "LCY"
header-img: "img/default.jpg"
tags:
    - pytorch
    - python
    - tensorflow
---

## Questions

大家跑实验时有没有遇到过这样的场景：时间赶得很急，要跑多组不同的实验。我往往是一口气堆满显存，等第二天看结果。然而很多实验，通常凌晨就跑完了，晚上的显卡性能就白白浪费了。如果自己写批处理脚本的话，也不知道第二天早上起来跑到哪里，管理起来很混乱。

因此，我搞了给开源项目，自动调度使用GPU的算法进程，规划实验进度。

## 项目地址

https://github.com/lichunown/gpu-limit.git

目前已经能够正常使用（调度算法目前是贪心策略，还有优化空间。）

目前也可以直接使用pip安装。

```bash
pip3 install gpulimit 
```

## 使用

具体使用，可以参考项目readme

https://github.com/lichunown/gpu-limit/blob/master/readme.md