# piper_ros Dockerfiles使用说明
在这里我使用了dockercompose.yml
这里就相当于一个自动化脚本，比如一些命名、权限之类的东西可以直接写在compose里面写好，相当于终端用了
## 关于piper_ros 包的拉取
为了避免5099极其sb的网络环境，在Dockerfile里面使用了
COPY 的方法将主机里面的已经下载好的源码直接拉到镜像里面
所以你要先查看以下Dockerfile里面COPY定义的路径，在这里把源码从github上clone下来之后，然后直接COPY进取
如果你对自己的网络环境**足够自信**的话，可以把COPY替换成
RUN git clone ....
## 那怎么用呢？
在docker同名的目录下，终端输入
docker compose build
就可以进行构建
## 关于容器如何读取宿主机的USB设备
在镜像制作容器的时候，可以通过
docker run --privileged -it --network=host 镜像名:镜像标签
的方式，通过privileged强行给容器升权
这个方法很笨，但是我能用，还有一个通过udev的方式，我还没研究，这个是最规范且效率最高的
你们有时间可以看看
# 这里面为什么没有Piper_SDK?
Piper_SDK

