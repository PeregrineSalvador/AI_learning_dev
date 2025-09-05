# Piper机械臂配置教程

## 应用体验提升

建议先安装一个**konsole**,用于美化终端，具有分屏灯高级功能，是对shell的部分功能的封装的更好的终端平台

打开shell ，输入：

```bash
sudo apt update && sudo apt install konsole
```

安装之后直接打开**konsole**，即可以使用了。

## Linux常识

笔者推荐很有必要先了解一下linux操作，有一定的Linux系统常识能让自己知道命令含义和用法。

这里给出一个Linux的常识介绍链接：[linux系统基本知识整理](https://blog.csdn.net/u011285477/article/details/90600501?ops_request_misc=&request_id=&biz_id=102&utm_term=linux常识&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-90600501.142^v102^control&spm=1018.2226.3001.4187)

## 安装ROS2 Humble官方教程
安装教程：https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html
这个文档就当作学习文档来看

## 安装ROS2 Humble的替代方案

这里可以有了一个替代方案，依次复制命令就行，一共四块命令：

```bash
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

```bash
sudo apt update
```

```bash
sudo apt install ros-humble-desktop
```

```bash
sudo apt install ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-controller-manager
```

在确定安装好ros2之后，需要激活一下：

```bash
source /opt/ros/humble/setup.bash
```

为了检查是否成功激活了ros2，可以输入如下命令：

```bash
ros2 -h
```

就能查看到一些ros2的相关命令。

为了确保所有终端都能识别到ros2

**查看当前设置的ROS发行版**：在终端中输入以下命令，检查当前 `$ROS_DISTRO` 的值：

```bash
echo $ROS_DISTRO
```

如果输出为**空**或者**不是**你预期的ROS版本（例如 `humble`、`foxy`、`galactic` 等），你就需要手动设置它。

**手动设置ROS_DISTRO**：知道你安装的ROS 2具体版本（例如 `humble`）后，使用以下命令设置环境变量（请将 `humble` 替换为你的实际ROS版本）：

```bash
export ROS_DISTRO=humble  # 将 'humble' 替换为你的 ROS 2 发行版名称
```

**注意**：这样设置的环境变量只在当前终端会话中有效。为了永久生效，你可以将上述命令添加到你的 `~/.bashrc` 文件中：

```bash
echo "export ROS_DISTRO=humble" >> ~/.bashrc  # 同样，替换 'humble' 为你的版本
source ~/.bashrc
```

## 安装ROS2编译器

完成ros2_humble的安装后，还需要安装ROS2的编译器，即安装colcon

```bash
source /opt/ros/humble/setup.bash
sudo apt update
sudo apt install python3-colcon-common-extensions
```

最后验证colcon安装成功：

```bash
colcon version-check
```

如果有正常输出，说明colcon安装成功

## 安装piper环境

按照piper官方手册安装piper环境
piper手册网址：https://github.com/agilexrobotics/piper_ros/tree/humble

正式使用之前，可以再装一个moveit，仿真好用(**网络跨境连接**之后下载速度会提升10倍)

```bash
sudo apt install ros-humble-moveit
```

安装完看下面的科研教程，学习Moveit2：
https://www.bilibili.com/video/BV1owtfe9EJH/?vd_source=36d67ceba7e90f1135a8f40fea37790f
如果读者已经对ros有基本认知，直接跳转到34分钟起看实际操作，但是如果读者对于ros不甚了解，建议1h视频认真看完。这个系列第一期质量也非常高，建立对ros的基本认知。尽可能快的完成理论学习，尽量早的能学习实践环节。

如果读者想直接使用moveit的话，这里给出：

```bash
ros2 launch piper_with_gripper_moveit demo.launch.py
```
### Piper使用注意事项
> 上电后航空插口如果插不紧，虽然能连上CAN，并且能失能，但是无法控制。要重新插紧

## 主从控制

必须严格按照assert中的进行接线



<img src="https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/wire_connection.PNG?raw=true" alt="wire_connection.PNG" style="zoom:50%;" />

CAN配置那一套是固定的，照常运行

from piper_sdk import *

先给主机械臂上电，然后运行程序

C_PiperInterface(can_name='can0', judge_flag=True).MasterSlaveConfig(0xFA, 0, 0, 0)

然后给主机械臂断电

随后给从机械臂上电
C_PiperInterface(can_name='can0', judge_flag=True).MasterSlaveConfig(0xFC, 0, 0, 0)

最后把两个机械臂都插上电。就可以进入主从示教
