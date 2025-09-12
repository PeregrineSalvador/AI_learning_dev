# 使用humble全镜像
FROM osrf/ros:humble-desktop-full

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=humble \
    PATH="/sbin:/usr/sbin:$PATH"

# 安装基础工具和USB相关工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    lsb-release \
    python3-pip \
    git \
    software-properties-common \
    sudo \
    iproute2 \
    net-tools \
    can-utils \
    ethtool \
    usbutils \  
    udev \
    && rm -rf /var/lib/apt/lists/*

# 创建piper用户并添加到必要的组
RUN groupadd piper && \
    useradd --create-home --shell /bin/bash --gid piper piper && \
    usermod -aG sudo,dialout,video,plugdev piper && \
    echo "piper ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    mkdir -p /workspace && chown piper:piper /workspace

# 安装Python依赖
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir python-can scipy piper_sdk

# 复制代码（使用用户名代替UID）
COPY --chown=piper:piper ./piper_ros /workspace/piper_ws/src/piper_ros
COPY --chown=piper:piper ./piper_SDK /workspace/piper_ws/src/piper_SDK
# 配置环境
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/piper/.bashrc && \
    echo "source /workspace/piper_ws/install/setup.bash" >> /home/piper/.bashrc && \
    git config --global --add safe.directory /workspace/piper_ws/src/piper_ros

# 创建USB设备目录并设置权限
RUN mkdir -p /dev/bus/usb && \
    chmod 777 /dev/bus/usb

# 设置工作目录和用户
WORKDIR /workspace/piper_ws
USER piper

# 初始化命令
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && /bin/bash"]
