## 镜像容器与DockerHub
### 镜像
- 类，从dockerfile构建，🉑指定标签
- 不推荐从docker-compose.yaml构建
- 更改永久储存

### 容器
- 实例，可从一个镜像生成多个实例
- docker run 直接生成的容器名字是任意的，并且不会自动删除
- **同一个名字的容器只能有一个**
- 进入容器命令
- 有正在运行与停止运行两种情况，只有停止运行的容器才能删除


### 常用指令
```bash
docker ps -a #用于列出所有的容器，包括正在运行的和已经停止的容器。
```
```bash
docker images #用于列出本地主机上所有的 Docker 镜像。
```
```bash
docker run -it 容器名 bash/fish
```

这个命令用于从指定的镜像创建并启动一个新的容器，并同时进入该容器的交互式终端。

- `-i`：以交互模式运行容器，保持标准输入流（stdin）打开。
- `-t`：分配一个伪终端，允许你与容器进行交互。
- `容器名`：这里需要替换为您要运行的镜像的实际名称。
- `bash/fish`：指定你希望在容器内执行的命令，通常是交互式的 shell，如 `bash` 或 `fish`。

```bash
docker exec -it # 表示进入容器
```

注意：`dockerhub`如果拉不了镜像，检查梯子和是否登录

### dockerfile构建镜像
#### 构建指令
```bash
docker build -f ./Docker -t custom -image .
```
`-t `指定标签
`-f` 指定`dockerfile`路径，如果是默认路径可以省略
最后的.表示构建上下文与copy路径🈶关系

#### 构建形式
- 每次构件都有缓存
- 会从上次缓存改变的地方开始构建
- 把测试的东西放在最后可以显著加快构建速度

#### FROM命令
- 指定基础镜像源
- 🉑以为本地或者远端镜像

#### ARG命令
- 设定dockerfile全局变量
- 用`${PASH}}`访问已经声明的变量

#### WORKDIR命令
- 指定之后的所有路径指令

#### RUN命令
- 默认用sh，不是bash
- 在`dockerfile`中的`source bash`命令比较麻烦，一般使用` ./install/setup.sh`代替
- 一个RUN命令是一层缓存，上下文不能互通
- 使用`apt-get update`也有可能不缓存
- 一般采用 \＋&& 实现在同一个RUN实现多条命令
```bash
RUN. /opt/ros/humble/setup.sh && cd /ros ws && colcon build --packages- select serial \ &&. /install/setup.sh && colcon build-cmake-args -DCMAKE BUILD TYPE=Release
```
注意：在忘记＋\或者&&会产生奇怪的报错

#### USER命令
- 指定接下去操作的用户，和权限相关

#### COPY命令
- 将本地文件复制进dockerfile
- 在本地文件发生改变时会认为本层变更

#### docker compose
- 从docker compose启动关闭容器
- 需要和docker compose.yaml在同一个文件夹路径下时使用
- docker compose up -d ＃-d表示在后台运行
- docker compose down

#### 服务名与用户名
- 不冲突就行

#### image

#### user

#### cmd
- 打开容器默认执行的任务

#### Ponrtainer可视化控制台

#### 系统维护命令
```bash
docker image prune # 清理悬空镜像
```

```bash
docker system prune # 全局清理
```

```bash
docker builder prune # 构建缓存清理
```