# Docker工作流说明

这么说，Docker工作流的核心的Dockerfile

而Dockerfile可以理解成为一个自动执行脚本

### 环境预创建

> 关于换源

建议在docker配置中将源换毫秒源

```
Processing triggers for libc-bin (2.35-0ubuntu3.10) ...
{
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://docker.m.daocloud.io",
    "https://docker.1panel.live",
    "https://hub-mirror.c.163.com",
    "https://mirror.baiduce.com"
  ]
}
```

> 创建一个新的Dockerfile
>
> Dockerfile不要拼错，它的目的是让你先自己创建一个**镜像**

创建过程如下：

1. **引用最基础的环境配置**

至于什么是“最基础的环境配置”，要根据你的实际情况来。

目前我推荐的就是要保证有ROS:humble，python，cmake，gcc，VScode相关内容

如果你想用别人的基础镜像，而不是自己完全从0开始的话 ，那你**首先需要从毫秒源上找到你想要的镜像的确切名称**

可能你想应用ROS:humble的基础镜像，而不是自己从0下载ros2，那你**不应该直接输入**：

```dockerfile
FROM ROS:humble
```

因为不一定在毫秒源上就叫这个名字，可能叫别的，你得先去毫秒源官网找到它。

2. **配置基础信息**

用户信息 初始化 权限 设置用户和环境变量 默认启动bash

> 注意，这里的创建过程没有给出明确程序，笔者目前探索不多。
>
> 大家可以自行探索一下，多阅读别人的Dockerfile用了什么

3. **创建自己的镜像**

写完dockerfile之后，在dockerfile目录下使用

```
docker build -t <镜像名称>:<标签> <构建上下文路径>
```

例如:

```0
docker build -t my-image:latest .
```

即可完成构建

### 镜像打包

1. 在终端进入你的容器（docker基础使用中讲过），验证你的环境安装没问题
2. 肆意地在你的镜像里面拉屎，直到跑通位置。但是要注意！在拉屎的过程中一定要记录下你安装了什么环境，做了什么操作。
3. 然后把这些操作整理好后，全部记录在宿主机的Dockerfile里面

==额外注意==

你在构建过程中，可以把创建文件夹，gitclone仓库等操作全部记录下来

但是尽量不要把编译命令也添加进去dockerfile，尽量让用户自己去编译。

**此时，Dockerfile就有了复现你所有工作的能力**

### 镜像验证

1. 将上述**完备的Dockerfile**再次打包成为一个镜像
2. 自己打开自己创建的镜像，然后验证功能是否完备

### 镜像提交

1. 验证完之后将镜像提交到云端（目前认为Dockerhub是最好的选择，免费）

docker的镜像动辄好几个G，并不建议直接传播

**或者**

2. 将Dockerfile提交到github，然后在里面附上自己的签名，别人可以直接用硬盘去copy你的镜像