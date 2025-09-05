# Realsense D435

### 📖 官方开发资源

开发 D435 主要依赖 **Intel RealSense SDK 2.0 (librealsense)**，这是其核心 SDK

- **官方 GitHub 仓库**：https://github.com/IntelRealSense/librealsense这里可以获取最新的源代码、示例程序、文档以及社区讨论。
- **API 文档**：SDK 提供了丰富的 API 用于获取深度、彩色、红外图像流，以及处理点云、对齐帧、录制和回放 bag 文件等。详细的 API 文档通常包含在库的安装路径中，或者你也可以在线查阅。
- **官方论坛**：https://community.intel.com/t5/Intel-RealSense-SDK/bd-p/realsense遇到棘手的问题时可以在这里搜索或提问。

### 🖥️ Linux 系统驱动安装

在 Linux 系统中使用 D435，你需要安装 `librealsense2`驱动库。安装方式如下：`Ubuntu `系统

```bash
# 注册服务器的公钥
sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
# 将服务器添加到存储库列表
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
# 更新软件包列表并安装所需的包
sudo apt-get update
sudo apt-get install librealsense2-dkms librealsense2-utils librealsense2-dev librealsense2-dbg
```

- `librealsense2-utils`包含了像 `realsense-viewer`这样的实用工具。
- `librealsense2-dev`是开发所需的头文件和链接库。

**安装验证**：

安装完成后，连接相机，运行 

```bash
realsense-viewer
```

如果能正常看到深度图、RGB 图像等数据流，说明安装成功。

**注意**：如果遇到与 `UEFI Secure Boot`相关的问题（例如 `Required key not available`错误），需要在 BIOS 中暂时禁用 Secure Boot，或者按照系统提示注册 MOK (Machine Owner Key)。

###    开发语言

使用 Python 进行开发。首先确保安装了 Python 包装器

```bash
pip install pyrealsense2
```

python语言的实例代码位于 **librealsense/wrappers/python/examples**



###  问题&答案

#### Q1：使用`realsense-viewer`访问进入相机后无法访问深度相机

#### Answer：

使用如下超用户权限访问：

```bash
sudo realsense-viewer
```

或者**设置 udev 规则**

在 Linux 系统中，像 RealSense 相机这样的 USB 设备由操作系统内核统一管理。普通用户默认没有权限直接访问这些硬件设备。虽然您可能已经安装了驱动和 SDK，但缺少一步关键的配置：**设置 udev 规则**。

1. 安装 `udev`规则包：

   ```bash
   sudo apt-get install librealsense2-udev-rules
   ```

2. **重新插拔 D435 相机 USB 线**：

3. ```bash
   realsense-viewer
   ```

   访问即可



#### 问题延申1

若按照步骤2出现如下返回消息：

```bash
 正在读取软件包列表... 完成
正在分析软件包的依赖关系树... 完成
正在读取状态信息... 完成                 
librealsense2-udev-rules 已经是最新版 (2.56.5-0~realsense.17054)。
librealsense2-udev-rules 已设置为手动安装。
下列软件包是自动安装的并且现在不需要了：
  libffi7
使用'sudo apt autoremove'来卸载它(它们)。
升级了 0 个软件包，新安装了 0 个软件包，要卸载 0 个软件包，有 0 个软件包未被升级。
```

可能是因为会遇到 **多个或冲突的 udev 规则文件**，或者规则尚未完全生效。

#### 解决方案如下

```bash
ls /etc/udev/rules.d/ | grep -i realsense
```

可能会看到类似下面的文件：

- `60-librealsense2-udev-rules.rules`(通常由较新版本的 `librealsense2-udev-rules`包提供)
- `99-realsense-libusb.rules`(可能由旧版 SDK 或源码编译安装时手动添加)

- **保留一个**：通常建议保留版本较新或由官方包管理的那个（如 `60-librealsense2-udev-rules.rules`）。

- **删除多余的**：如果确认 `99-realsense-libusb.rules`是旧版本遗留的，可以删除它（**建议先备份**）：

  ```bash
  sudo rm /etc/udev/rules.d/99-realsense-libusb.rules
  ```

更新或删除规则文件后，**必须让 udev 重新加载配置才能生效**。

🔄 **重新加载 udev 规则并触发：**

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

或者，你也可以选择重启 udev 服务（部分系统可能需要）：

```bash
sudo systemctl restart systemd-udevd.service
```

**完成上述任何更改后，最关键的一步是：重新插拔你的 D435 相机 USB 线。**



#### 问题延申1.1

```bash
ls /etc/udev/rules.d/ | grep -i realsense
```

没有输出时，可能是在 `/etc/udev/rules.d/`目录下，**没有文件名包含 "realsense"（不区分大小写）的规则文件**。按如下方法解决：

### 🔍 排查步骤

```bash
ls -l /etc/udev/rules.d/ #直接查看 /etc/udev/rules.d/目录的所有文件
```

```bash
ls -l /usr/lib/udev/rules.d/ | grep -i realsense
```

查找目录中的找到类似于 `60-librealsense2-udev-rules.rules`或 `99-realsense-libusb.rules`的文件。（假设在 `/lib/udev/rules.d/`中找到）

将 `/lib/udev/rules.d/`目录下的规则文件链接到 `/etc/udev/rules.d/`目录：

```bash
sudo ln -s /lib/udev/rules.d/60-librealsense2-udev-rules.rules /etc/udev/rules.d/
```

**重新加载 udev 规则并触发**：

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

**重新插拔你的 D435 相机 USB 线**

**如果规则文件确实不存在**：重新安装包后，`/etc/udev/rules.d/`目录下依然没有对应的规则文件，那可能是包本身的问题。你可以尝试**手动创建规则文件**。使用以下命令创建并编辑一个规则文件

```bash
sudo touch /etc/udev/rules.d/99-realsense-libusb.rules
```

之后，需要**手动编辑此文件**，将正确的规则内容（通常可以在 Librealsense 的源码中的 `scripts`目录找到）添加进去，保存后**重新加载 udev 规则**并**重新插拔相机**。
