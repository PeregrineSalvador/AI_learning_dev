## Docker（软件部署技术）

参考视频：[https://www.bilibili.com/video/BV1THKyzBER6?t=2366.9](https://www.bilibili.com/video/BV1THKyzBER6?t=2366.9)


### 三个概念：容器，镜像，镜像仓库（`Dockerhub`）

#### 容器

使用容器化技术给应用程序封装独立的运行环境；我们把每一个运行环境称作一个**容器**。运行容器的计算机被称为宿主机。

Docker容器 VS虚拟机

Docker容器共用一个系统内核；而每一个虚拟机都包含一个操作系统完整的虚拟内核；所以Docker更轻便

#### 镜像

镜像是容器的模板，类似于软件安装包；则容器就类似于软件。

#### 镜像仓库

存放镜像的地方，官方仓库为`Docker Hub`,类似于`github`这样的社区

### Docker 安装

#### Linux系统

在终端执行以下命令：

```bash
curl -fsSL https://get.docker.com -o install-docker.sh
```

```BASH
sudo sh install-docker.sh
```

#### Windows系统

在任务栏搜索“功能”，点击“启用或关闭Windows功能”，勾选“Virtual Machine Platform”（对应中文为虚拟机平台），勾选“适用于Linux的Windows子系统”（即WSL）

之后按照提示重启电脑。

重启电脑后，在终端输入如下命令：

```shell
wsl --set-default-version 2  # 设置WSL默认版本
```

```shell
wsl --update --web-download
```

之后进入如下网址：https://www.docker.com/

点击“Download Docker Desktop”，选择"Download for Windows - AMD64"

后续点击下载的.exe文件安装即可。

### Docker命令（Linux系统）

#### `Docker pull`：用来从仓库下载镜像，示例代码如下：

```bash
sudo ducker pull docker.io/library/ImagesName:latest
```

`docker.io`：docker仓库的注册表地址；此注册表地址为官方地址，官方地址可忽略；
`library`：命名空间（作者名）；此命名空间为官方命名空间，官方命名空间可忽略；
`ImagesName`：镜像的名称；
`:latest`：版本号，最新版本号可不写。

```bash
sudo docker pull -platfrom=xxxxxx container ID & name #表示拉取nginx特定的CPU架构镜像
```

​	因为docker的镜像在不同版本下会有不同的CPU架构版本，所以对于一些例如香橙派这种迷你主机运行docker时，就需要确认与其匹配的镜像架构；对于普通的主机则会自动匹配，无需关注。

#### `Docker images`：显示所有已下载的镜像信息，示例代码如下：

```bash
sudo docker images
```

#### `Docker rmi`：删除已下载的镜像，示例代码如下：

```bash
sudo docker rmi container ID & name # 删除ImagesID对应的镜像，ImagesID可通过ducker images查看，ImagesID可换成ImagesName。
```
#### `Docker rm`：删除容器，示例代码如下：

```bash
sudo docker rm container ID & name # 删除容器。
```

```bash
sudo docker rm -f container ID & name # 强制删除正在运行的容器。
```

#### `Docker ps`：查看容器简单的信息，示例代码如下：

```bash
sudo docker ps # 查看正在运行的容器
```

```bash
sudo docker ps -a # 查看所用已经建立的容器
```

简单信息包括：container ID，images，command，created，status，port

#### `Docker run`：使用镜像创建并运行容器（**重点**），示例代码如下：

```bash
sudo docker run container ID & name
```

##### -d 参数

```bash
sudo docker run -d container ID & name #与上述的区别是设置成分离模式，使得容器在后台运行，不会阻塞当前窗口。
```

**注意：**使用`docker run`时，若发现本地不存在镜像，则会自动拉取并创建容器，默认的拉取地址是官方库，若要从指定库拉取，则可看如下实例代码：

```bash
sudo docker run registry.cn-hangzhou.aliyuncs.com/namespace/image:tag
```

当执行上述命令时，如果本地没有该镜像，Docker 会自动从指定的阿里云仓库拉取，而不是 Docker Hub 官方库。

​	故此处使用时，前面的`docker pull`可以省略。

##### -p 参数

 若要访问docker的内部网络，则需要进行端口映射，实例代码如下：

```bash
sudo  docker run -p 80:80 container ID & name #前一个80时宿主机端口，后一个80是容器内的端口
```

挂载卷，由于我们在删除容器时，容器内的数据也会被删除，使用挂载卷，使得容器内的目录与宿主机的目录相互绑定，这样修改容器内目录的数据或宿主机目录中的数据时，对应另一方也会同步被修改，当容器被删除时，使用挂载卷的目录会在宿主机中保留。实例代码如下：

##### -v 参数

```bash
sudo docker run -v 宿主机目录:容器内目录 # 绑定挂载
```

**注意：**使用绑定挂载时，宿主机的目录（暂时）会覆盖容器内的目录。

```bash
sudo docker volume create nginx_html # 创建一个新的挂载卷，nginx_html为卷的名字
sudo docker run nginx_html:容器内目录 # 命名卷挂载
```

```bash
sudo docker volume inspect nginx_html #查看命名卷目录等详细信息
```

查看目录下的内容：

```bash
sudo -i # 切换为root用户
cd 目录路径 # 转到命名卷的目录下
vi index. # 查看目录内容
```

**注意：**命名卷的特殊功能，他会默认将容器内的目录下内容复制到该命名卷目录下，绑定挂载则没有这种功能。

其他与挂载卷的相关命令：

```bash
sudo docker volume list # 列出所有创建过的卷
sudo docker volume rm 挂载卷名称 # 删除一个卷
sudo docker volume prune -a # 删除所有没有任何容器在使用的卷
```

##### -e 参数

向容器中传递环境变量

##### --name参数

给容器自定义名称，该名称在宿主机上是唯一的不能重复，实例代码如下：

```bash
sudo docker run -d --name name
```

##### -it参数

让控制台进入容器进行交互

##### --rm参数

当容器停止时就删除容器

**注意：**一般情况下，-it与--rm通常一起使用，用于临时调整一个容器。

##### --restart参数

`--restart always`只要容器停止就立即重启；
`--restart unless-stopped`除手动停止的容器情况外，只要容器停止就立即重启。

#### `Docker stop`：停止运行容器，示例代码如下：

```bash
sudo docker stop container ID & name
```
#### `Docker start`：启动运行容器，示例代码如下：

```bash
sudo docker start container ID & name
```

**注意：**启动之后，上次启动的参数均已保存，无需重新配置。

#### `Docker inspect`：查看容器配置信息，示例代码如下：

```bash
sudo docker inspect container ID & name
```

结果会打印一长串内容，我们可以将打印信息喂给AI，进行相关询问。

#### `Docker create`：只创建容器，不启动，示例代码如下：

```bash
sudo docker create container ID & name
```

#### `Docker logs`：查看容器日志，示例代码如下：

```bash
sudo docker logs container ID & name -f  # -f表示滚动查看
```

### Docker的技术原理

Docker允许开发者将应用及其依赖打包到一个轻量级的、可移植的容器中，然后发布到任何支持Docker的Linux主机上。Docker容器在运行时，与其他容器相互隔离，共享同一操作系统内核，但拥有自己的文件系统、网络配置、进程空间等。这种隔离是通过Linux的Namespace和Cgroup两大机制实现的。

简单来说，每一个docker内部都是一个独立的运行环境，都有独立的Linux系统。`docker exec`可以在容器内部执行命令。

```bash
sudo docker exec container ID & name Linux命令 # 查看进程为 ps -ef
```

```bash
sudo docker exec -it container ID & name /bin/sh # 进入一个正在运行的容器内部，获得一个交互式的命令行环境，可进入容器内部执行linux命令，查看系统文件等等操作。
```

**注意：**由于容器内的系统是极简的，所以当运行报错是时，关注是否存在相关命令未安装的情况。

**解决方法：**

步骤1：检查容器内部系统发行版

```bash
cat /etc/os-release # 执行sudo docker exec -it container ID & name /bin/sh之后操作
```

步骤2：根据发行版，选择合适的安装方式，具体安装方式可询问AI

### Dockerfile

下面的例子用于如何制作一个镜像，并将其推送到`Docker Hub`上：

步骤一：在自己电脑上制作好自己的项目文件

步骤二：在项目文件夹中创建`Dockerfile`文件，注意：D要大写，文件没有后缀名；文件中写如下代码：

```dockerfile
FROM python:3.13-slim # python:3.13-slim 为基础镜像，可在dockerHub上搜索。

WORKDIR /app # /app为镜像的工作目录，后续工作均在此目录下进行。

COPY . . # 第一个.代表宿主机的当前目录，第二个点代表镜像中的工作目录。

RUN install -r requirements.txt # 在镜像中安装依赖，requirements.txt是自己配置编写的

EXPOSE 8000 # 生声明镜像服务端口，起提示为准，不写不影响

CMD ["python3","main.py"] # 容器启动时的默认执行命令，这里为"python3"和"main.py",一个Dockerfile文件中只能有一个CMD或者ENTRYPOINT，其中ENTRYPOINT优先级更高，不易被覆盖。
```

步骤三：构建镜像，在当前目录下执行：

```python
docker build -t 镜像的名称 . # .表示在当前目录下构建
```

步骤四：基于这个镜像创建一个容器并且运行，执行：

```python
docker run -d -p 8000:8000 镜像的名称
```

步骤五：推送到`DockerHub`上。执行：

```python
docker login
```

打开网站，填写验证码。

当命令行显示Login Succeeded，则表示成功。

之后新建一个终端，重新打开一个镜像：

```python
docker build -t DockerHub的用户名/镜像的名称 .
```

```python
docker push DockerHub的用户名/镜像的名称
```

说明：`DockerHub`需要有自己的账户，执行步骤1，2，5即可完成；步骤3，4为学习步骤。

### Docker网络

默认使用桥接模式。

```bash
sudo docker network create network1 # 创建一个子网network1
```

```bash
sudo docker run -d \
--name my_mongodb
-e MONGO_INITDB_ROOT_USERNAME=Teaven \
-e MONGO_INITDB_ROOT_PASSWORD=123456 \
--network network1 \
mongo
```

```bash
sudo docker run -d \
--name my_mongodb_express
-p 8081:8081
-e ME_CONFIG_MONGODB_SERVER=my_mongodb \
-e ME_CONFIG_MONGODB_ADMINUSERNAME=Teaven \
-e ME_CONFIG_MONGODB_ADMINUSERPASSWORD=123456 \
--network network1 \
mongo_express
```

`--name`指定容器名称；`-e`设置环境变量，设置用户名与密码，默认账号是admin，密码是pass；`--network network1`：将此容器连接到之前创建的名为`network1`的Docker网络。这样容器可以与同一网络中的其他容器互相通信。

### Docker compose

容器编排技术，使用`.yaml`文件管理多个容器

Docker compose文件可以理解成一个或多个Docker run命令，按照特定格式列到一个文件中，上述Docker网络所举事例的`docker-compose.yaml`文件中内容如下：(AI关键词：生成一个等价的docker compose文件)

```yaml
version: '3.8'  # 指定 Docker Compose 文件的版本

services:
  mongo:
    image: mongo  # 使用 MongoDB 官方镜像
    container_name: my_mongodb  # 设置容器名称
    environment:
      MONGO_INITDB_ROOT_USERNAME: Teaven  # 设置根用户名
      MONGO_INITDB_ROOT_PASSWORD: 123456  # 设置根密码
    networks:
      - network1  # 连接到指定网络

  mongo_express:
    image: mongo_express  # 使用 Mongo Express 镜像
    container_name: my_mongodb_express  # 设置容器名称
    ports:
      - "8081:8081"  # 映射端口
    environment:
      ME_CONFIG_MONGODB_SERVER: my_mongodb  # 指定 MongoDB 服务器
      ME_CONFIG_MONGODB_ADMINUSERNAME: Teaven  # 设置管理用户名
      ME_CONFIG_MONGODB_ADMINUSERPASSWORD: 123456  # 设置管理密码
    networks:
      - network1  # 连接到同一网络

networks:
  network1:  # 创建自定义网络
```

```bash
sudo docker compose up -d # 启动文件中定义的所有容器，但在已经启动容器后执行，则没有效果
```

```bash
sudo docker compose down # 停止并删除容器
```

```bash
sudo docker compose stop # 停止容器
```

```bash
sudo docker compose start # 启动容器
```

**注意：**执行`sudo docker compose up -d`必须是`.yaml`标准文件名，即`docker-compose.yaml`；否则须执行：

```bash
sudo docker compose -f 非标准文件名（可在其他目录下） up -d
```
