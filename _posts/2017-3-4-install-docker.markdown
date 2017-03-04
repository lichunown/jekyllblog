---
layout:     post
title:      "windows下安装docker"
subtitle:   "不友好的win10家庭版"
date:       2017-02-21
author:     "LCY"
header-img: "img/default.jpg"
tags:
    - docker
---

安装docker，真是折腾死我了。
**如果你的电脑也是win10家庭版，请不要直接装docker。**
windows版docker依赖`Microsoft Hyper-V `

然而，这一功能只有在windows10 专业版上才有。面临升级的808￥，我不禁望而却步……（本来真的想交钱，然而发现没有visa银行卡。。GG）

当然了，如果你幸运的是专业版，可以直接安装docker。[docker官方下载地址](https://download.docker.com/win/stable/InstallDocker.msi)

-----

所以，我只能试一试plan B，安装docker ToolBox。

#### 下载docker ToolBox
[docker ToolBox 官方下载地址](https://github.com/docker/toolbox/releases/download/v1.12.5/DockerToolbox-1.12.5.exe)

#### 安装

按照默认的配置装就好了。

#### 然后GG

**官方教程是双击`Docker Quickstart Terminal`
![Docker Quickstart Terminal](img/in-post/docker-install/quickstartBtn.png)
**然而，报错！！！**

#### 各种测试。。。

GG之后，开启瞎倒腾之旅。

- **重装（修复）virtualbox**

  报错信息是关于Oracle VM VirtualBox的，所以很自然，用安装目录备份的virtualBox安装文件修复一下。如果你是默认安装目录的话，大概是在`C:\Program Files\Docker Toolbox\installers\virtualbox\virtualbox.msi`里。

- **重启**

  MDZZ，还是祭出重启大法。

- **打开virtualbox，检查default是否存在**

  重启之前的报错信息有提示无法找到虚拟机的IP，因此我特地打开虚拟机看了眼，default还是存在的。

- **各种命令测试**

  显示一下docker的信息吧。

  ```bash
  $ docker-machine ls
  ```
  输出：
  ```bash
  NAME      ACTIVE   DRIVER       STATE   URL   SWARM   DOCKER    ERRORS
  default   -        virtualbox   Saved                 Unknown
  ```
  大概是没什么问题，启动看看

  ```bash
  $ docker-machine start

  Starting "default"...
  (default) Waiting for an IP...
  Machine "default" was started.
  Waiting for SSH to be available...
  Detecting the provisioner...
  Started machines may have new IP addresses. You may need to re-run the `docker-machine env` command.
  ```

  好像不行，按提示搞吧。
  ```bash
  $ docker-machine env

  Error checking TLS connection: Error checking and/or regenerating the certs: There was an error validating certificates for host "192.168.99.100:2376": x509: certificate signed by unknown authority
  You can attempt to regenerate them using 'docker-machine regenerate-certs [name]'.
  Be advised that this will trigger a Docker daemon restart which will stop running containers.
  ```
  继续按提示来
  ```bash
  $ docker-machine regenerate-certs default

  Regenerate TLS machine certs?  Warning: this is irreversible. (y/n): y
  Regenerating TLS certificates
  Waiting for SSH to be available...
  Detecting the provisioner...
  Copying certs to the local machine directory...
  Copying certs to the remote machine...
  Setting Docker configuration on the remote daemon...
  ```

  **再试一次**
  ```bash
  $ docker-machine env
  SET DOCKER_TLS_VERIFY=1
  SET DOCKER_HOST=tcp://192.168.99.100:2376
  SET DOCKER_CERT_PATH=C:\Users\lichunyang\.docker\machine\machines\default
  SET DOCKER_MACHINE_NAME=default
  REM Run this command to configure your shell:
  REM     @FOR /f "tokens=*" %i IN ('docker-machine env') DO @%i
  ```
  **似乎可以了，复制上一句`@FOR /f "tokens=*" %i IN ('docker-machine env') DO @%i`执行**

  ```bash
  $ @FOR /f "tokens=*" %i IN ('docker-machine env') DO @%i
  ```
  ```bash
  $ docker-machine ssh
                          ##         .
                  ## ## ##        ==
               ## ## ## ## ##    ===
           /"""""""""""""""""\___/ ===
      ~~~ {~~ ~~~~ ~~~ ~~~~ ~~~ ~ /  ===- ~~~
           \______ o           __/
             \    \         __/
              \____\_______/
   _                 _   ____     _            _
  | |__   ___   ___ | |_|___ \ __| | ___   ___| | _____ _ __
  | '_ \ / _ \ / _ \| __| __) / _` |/ _ \ / __| |/ / _ \ '__|
  | |_) | (_) | (_) | |_ / __/ (_| | (_) | (__|   <  __/ |
  |_.__/ \___/ \___/ \__|_____\__,_|\___/ \___|_|\_\___|_|
  Boot2Docker version 1.12.6, build HEAD : 5ab2289 - Wed Jan 11 03:20:40 UTC 2017
  Docker version 1.12.6, build 78d1802
  docker@default:~$
  ```





### 应该可以用了吧……真是身心俱疲



## 后来

**如果报错是**

```bash
$ docker run hello-world
docker: An error occurred trying to connect: Post http://%2F%2F.%2Fpipe%2Fdocker_engine/v1.24/containers/create: open //./pipe/docker_engine: The system cannot find the file specified..
See 'docker run --help'.
```

**那就执行**

```bash
$ docker-machine env

SET DOCKER_TLS_VERIFY=1
SET DOCKER_HOST=tcp://192.168.99.100:2376
SET DOCKER_CERT_PATH=C:\Users\lichunyang\.docker\machine\machines\default
SET DOCKER_MACHINE_NAME=default
REM Run this command to configure your shell:
REM     @FOR /f "tokens=*" %i IN ('docker-machine env') DO @%i

$ @FOR /f "tokens=*" %i IN ('docker-machine env') DO @%i
```
其中`@FOR /f "tokens=*" %i IN ('docker-machine env') DO @%i`是复制自`docker-machine env`执行结果的最后一句。

**然后差不多就能用了**。



-----

然而，作为一个shadowsocks重度用户。。。

```bash
$ docker run hello-world
docker: An error occurred trying to connect: Post https://192.168.99.100:2376/v1.24/containers/create: http: error connecting to proxy http://127.0.0.1:1080: dial tcp 127.0.0.1:1080: connectex: No connection could be made because the target machine actively refused it..
See 'docker run --help'.
```

关了代理吧……

```bash
$ set http_proxy=
```

MDZZ, 连不上网了……

```bash
$ docker run hello-world
Unable to find image 'hello-world:latest' locally
Pulling repository docker.io/library/hello-world
docker: Network timed out while trying to connect to https://index.docker.io/v1/repositories/library/hello-world/images. You may want to check your internet connection or if you are behind a proxy..
See 'docker run --help'.
```



**怎么办……**

1. ssh连接docker（或者直接通过virtualBox登录）

2. 在 `/var/lib/boot2docker/profile` 添加如下内容。开启代理

   ```
   export "HTTP_PROXY=http://192.168.56.1:1080"
   export "HTTPS_PROXY=http://192.168.56.1:1080"
   ```
   我是使用的shadowsocks。至于IP
   ![ip查看](img/in-post/docker-install/ip1.png)
   可以看到，有两个虚拟的。至于用哪个，`ping`一下，哪个通用哪个呗。
