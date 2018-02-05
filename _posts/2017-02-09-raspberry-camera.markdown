---
layout:     post
title:      "树莓派摄像头的几种使用方法"
subtitle:   "以前的笔记，排版很乱"
date:       2017-02-09
author:     "LCY"
header-img: "img/default.jpg"
tags:
    - raspberry
    - 嵌入式
---


首先，开启摄像头功能。命令行输入
```
sudo raspi-config
```

选择*Enable Camera*
确保树莓派开启摄像头功能。

- 拍照

```
raspistill -o a.jpg -rot 180 # rot 是旋转180度
```

```
# raspistill 参数
# -v：调试信息查看。
# -w：图像宽度
# -h：图像高度
# -rot：图像旋转角度，只支持 0、90、180、270 度
# -o：图像输出地址，例如image.jpg，如果文件名为“-”，将输出发送至标准输出设备
# -t:获取图像前等待时间，默认为5000，即5秒
# -tl：多久执行一次图像抓取。
```

eg：

```
raspistill -o image%d.jpg -rot 180 -w 1024 -h 768 -t 20000 -tl 5000 -v
# 截取一张宽1024px，高768px，旋转180度的图片，抓取的总时长为20秒，
# 并且每5秒抓取一张，保存的文件名为image1.jpg,image2.jpg以此类推。
```

- 视频

> (安装mplayer)[http://www.softpedia.com/get/Multimedia/Video/Video-Players/MPlayer-for-Windows-Full-Package.shtml]

> 安装netcat

> 本地

```
nc -L -p 5001 | mplayer -fps 25 -cache 1024 -
```
> raspberry

```
raspivid -t 99999 -w 1280 -h 800 -o - | nc IP 5001 
# -t 0 实时输出

```

- web 视频

1. 安装**ffmpeg**

```
raspivid -t 0 -w 320 -h 240 -o - | ffmpeg -i - -s 320x240 -f mpeg1video -b 800k -r 30 http://127.0.0.1:8082/yourpassword
```

1. 创建 Node server 监听 http://127.0.0.1:8082

```
git clone https://github.com/phoboslab/jsmpeg.git webcam
cd webcam && node stream-server.js yourpassword
# Listening for MPEG Stream on http://127.0.0.1:8082/<secret>/<width>/<height>
# Awaiting WebSocket connections on ws://127.0.0.1:8084/
```

然后打开stream-example.html就可以看到实时监控画面了，如果是远程调试需要稍作更改：

```
# edit stream-example.html
var client = new WebSocket( 'ws://RASPI_LOCAL_IP:8084/' );
        var player = new jsmpeg(client, {canvas:canvas});

# @/path/to/webcam
python -m SimpleHTTPServer 8080

# in your browser
http://RASPI_LOCAL_IP:8080
```

- 另一个方法

```
sudo apt-get update
sudo apt-get install vlc
sudo raspivid -o - -t 0 -w 640 -h 360 -fps 25|cvlc -vvv stream:///dev/stdin --sout '#standard{access=http,mux=ts,dst=:8090}\' :demux=h264
```

> 在电脑端，无论是Windows，Linux还是OSX，或者安卓机器，只要能安装VLC，现在就可以打开VLC，然后打开媒体
> 网络串流输入http://PI的IP地址:8090查看实时不卡的网络监控了。
第一行是更新软件数据库
第二行是安装vlc
第三行是使用PI官方的raspivid捕获视频工具把视频流输出到vlc，通过vlc转码成h264网络视频流通过http协议以ts的形式封装，然后输出到8090端口，用这个当监控只要网络稳定绝对不卡。
看到以下窗口就说明开始正常输出内容了。然后在其它VLC客户端打开网络串流就行了。
还有哦。摄像头模块工作的时候那个红色的灯会一直亮，嘿嘿，要想禁用它的话

```
sudo nano /boot/config.txt
```

> 然后加入一行

```
disable_camera_led=1
```

##  树莓派摄像头模块没有/dev/video0设备节点的问题 

>> 相信大家入手的树莓派都玩起来了，买了摄像头模块的也demo起来的网上都有的几个例子，但是真正自己写程序的时候就会出现各种各样的问题，我是用树莓派的camera来实现一个远程视频传输的过程，Camkit就是这么一个简单的例子，然而一切准备就绪之后运行起来就会发现找不到/dev/video0设备，但是树莓派官方自带的raspistill却能够用起来，这怎么回事呢？
先放个wiki的地址：
https://wiki.archlinux.org/index.php/Raspberry_Pi
这里讲了很多关于启动的问题，其中Raspberry Pi camera module这一章节就是说如何起camera。树莓派中的camera module是放在/boot/目录下以固件的形式加载的，不是一个标准的v4l2的摄像头ko驱动，所以加载起来之后会找不到/dev/video0的设备节点，这是因为这个驱动是在底层的，v4l2这个驱动框架还没有加载，所以要在/etc/下面的modules-load.d/rpi-camera.conf里面添加一行bcm2835-v4l2,这句话意思是在系统启动之后会加载这个文件中模块名，这个模块会在树莓派系统的/lib/modules/xxx/xxx/xxx下面，添加之后重启系统，就会在/dev/下面发现video0设备节点了。这个文件名可能不是叫modules-load.d/rpi-camera.conf，也有可能直接就是/etc/modules，我用的是树莓派2，就是/etc/modules。
OK，祝大家玩的愉快。

然后，就可以用*motion*

```
sudo apt-get install motion
```
输入命令编辑motion：
```
sudo nano /etc/default/motion
```
把里面的no修改成yes，让他可以一直在后台运行：
```
start_motion_daemon=yes
```
修改motion的配置文件 
输入命令：
```
sudo vim /etc/motion/motion.conf
```
将第11行的daemon off 改成daemon on
将stream_localhost on改成off

3.配置启动

（1）输入下面命令启动服务：
```
sudo service motion start  
```
（2）输入以下命令开启motion：
```
sudo motion
```