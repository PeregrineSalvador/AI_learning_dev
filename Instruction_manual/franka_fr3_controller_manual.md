![Franka](https://i-blog.csdnimg.cn/blog_migrate/23d69de441bee4ed751d1f4cf2376b45.jpeg#pic_center)  
本文主要分析如何使用[Franka机器][Franka]人C++代码库——[libfranka][]来进行运动生成与控制。包含机器人学相关知识、Franka机器人的特性以及笔者在使用过程中的一点心得体会。笔者基于libfranka 0.8.0 版本进行开发调试。除了编程技巧外，本文还将在一定程度上讨论Franka 机器人官方运动生成与阻抗控制方法的基本特征，以及一些实际使用技巧。重点在于介绍FCI手册中没有将的一些实用技术细节以及笔者对Franka机器人的分析。

#### 文章目录 

 *  [libfranka综述][libfranka 1]
 *  [运动生成][Link 1]
 *   *  [关节空间运动生成][Link 2]
     *  [笛卡尔空间运动生成][Link 3]
 *  [阻抗控制与力控][Link 4]
 *   *  [以力矩为输入的运动控制][Link 5]
     *  [外力估计与控制][Link 6]
     *  [阻抗控制][Link 7]
 *  [其它参考][Link 8]

这里假设读者已具备以下基础知识：

 *  C++基础
 *  Eigen基础
 *  机器人学基础

最新版libfranka可[从这里下载][Link 9]。编译过程[参考此文][Link 10]。

## libfranka综述 

![libfranka](https://i-blog.csdnimg.cn/blog_migrate/7e502d53efb687e141bd974fd0114f7f.png)  
libfranka是采用C++开发的工具包，用以开发Franka机器人的应用。libfranka提供了实时的控制机制，使能用户开发复杂的运动规划/控制算法，十分方便。Franka机器人的控制频率为 1kHz ，但是实际留给用户代码的执行时间 <300 μ \\mu μs。切记不要让代码过于冗长，也不要太多额外的外部读写操作。

本文进讨论Franka机器人的实时运动生成与控制部分，假设机器人未安装官方电爪。用户可以自行拆解机器人的电爪，并在Desk界面的Settings中将末端执行器设置为None。  
![control](https://i-blog.csdnimg.cn/blog_migrate/411ef21b10e00582556c5532bb8fd4b5.png)  
玩转franka机器人其实指的是掌握libfranka提供的运动生成策略以及通过直接控制关节力矩设计控制策略。如上图所示，实时控制信号为关节力矩，用户也可以实时指定关节位置及速度、笛卡尔空间位置及速度、肘部位置。libfranka非实时指令的内容比较简单易懂，本文不做专门介绍。

以下内容中，我们假设读者已经对franka::Robot::Control的工作机理有初步的了解，至少看过[FCI手册][Franka]或者[此文][Link 10]。

## 运动生成 

![rt](https://i-blog.csdnimg.cn/blog_migrate/27bbd6c3ec03442ce2ce4bb27d09477b.png)  
我们从官方例程说起。按照FCI手册的说法，如上图所示，用户给定的指令信号（下标为c，command）为关节空间位置、速度，笛卡尔空间位置、速度。注意运动信号务必要“连续”，即变化不能超限，[限制阈值参考FCI手册][FCI]。随后，机器人控制器根据运动学特性换算出期望信号（下标为d，desired）。也就是说，实际控制机器人的其实是期望信号。

### 关节空间运动生成 

关节空间的运动生成不需要运动学计算，[FCI手册][FCI 1]上这样说：

> When you are using a joint motion generator, the Robot kinematics completion block will not modify the commanded joint values and therefore  q d , q ˙ d , q ¨ d q\_d, \\dot\{q\}\_d, \\ddot\{q\}\_d qd,q˙d,q¨d and  q c , q ˙ c , q ¨ c q\_c, \\dot\{q\}\_c, \\ddot\{q\}\_c qc,q˙c,q¨c are equivalent. Note that you will only find the  d d d signals in the robot state.

也就是说  q d = q c q\_\{d\} = q\_\{c\} qd=qc， q ˙ d = q ˙ c \\dot\{q\}\_\{d\} = \\dot\{q\}\_\{c\} q˙d=q˙c。其实不是这么简单，我们先看官方例程：

```java
std::array<double, 7> initial_position;
double time = 0.0;
robot.control([&initial_position, &time](const franka::RobotState& robot_state,
                                             franka::Duration period) -> franka::JointPositions {
            
   
     
     
      time += period.toSec();
      if (time == 0.0) {
            
   
     
     
        initial_position = robot_state.q_d;
      }
      double delta_angle = M_PI / 8.0 * (1 - std::cos(M_PI / 2.5 * time));
      franka::JointPositions output = {
            
   
     
     {
            
   
     
     initial_position[0], initial_position[1],
                                        initial_position[2], initial_position[3] + delta_angle,
                                        initial_position[4] + delta_angle, initial_position[5],
                                        initial_position[6] + delta_angle}};
      if (time >= 5.0) {
            
   
     
     
        std::cout << std::endl << "Finished motion, shutting down example" << std::endl;
        return franka::MotionFinished(output);
      }
  return output;
});
```

关于franka::Robot::control方法的使用读者请自行参考[frankaemika.github.io][]上的介绍，其它就是C++ 基础，此处不做赘述。该例程完成了关节4、5、7的运动规划。根据变量delta\_angle（此处记作  δ θ \\delta\\theta δθ）的公式：  
 δ θ = π 8 ( 1 − cos ⁡ ( 2 π 5 t ) ) \\delta\\theta = \\frac\{\\pi\}\{8\} \\left(1-\\cos\\left(\\frac\{2\\pi\}\{5\}t\\right)\\right) δθ=8π(1−cos(52πt)) 可以看出，这是一个S形曲线运动规划。使用正余弦函数可以保证轨迹在任何位置无穷阶可微，即保证了光滑性。此时只要限制振幅（此处为 π / 8 \\pi/8 π/8）就可以确保速度、加速度、加加速度不超限。

为了理解问题，我们简化程序：

```java
double delta_angle = M_PI / 64.0;      
franka::JointPositions output = {
            
   
     
     {
            
   
     
     initial_position[0], initial_position[1],                  
                                  initial_position[2], initial_position[3],
                                  initial_position[4], initial_position[5],
                                  initial_position[6] + delta_angle}};
```

给定一个固定值，结果会怎么样？我们读取7轴的期望信号与指令信号共同作图：  
![qcqd](https://i-blog.csdnimg.cn/blog_migrate/0cc21cadcaca71845db8f29116754803.jpeg#pic_center)  
有意思了，这酷似一个欠阻尼二阶控制系统的瞬态响应曲线。我们再把速度、和加速度曲线画出来：  
![dqd](https://i-blog.csdnimg.cn/blog_migrate/dc13cc771b719c0af8a77ec09c1f6fc5.jpeg#pic_center)  
![ddqd](https://i-blog.csdnimg.cn/blog_migrate/3b315e5e8a3f8ad91d5907781a1f287e.jpeg#pic_center)  
这下更明显了，开始时控制系统将加加速度设置为最大，进而将加速度加到最大，反复调节，最终系统稳定。注意 franka::Robot::control 函数参数中除了运动生成器外还包含一个控制器，如果不指定控制器，控制系统会调用内部的阻抗控制器实现。

那么，如果读取官方例程中指令信号与期望信号作比较，会得到什么样的结果呢？这个问题留给读者自行检验。（提示：一个周期内差的积分为0）

综上，我们可以看出一个有用的技巧是：不要直接给定阶跃控制目标。平滑过渡对于实际操作十分重要。原因或许是缺少一个内部插值机制，偏差过大会导致控制器输出达到峰值。

### 笛卡尔空间运动生成 

如果想要单纯输出\*\*位置（3维）而非位姿（6维）\*\*运动轨迹，那么可以参考[官方例程][Link 11]；否则时常会报出加速度过大的错误。因此笔者不建议采用这种方式生成笛卡尔空间位姿轨迹。官方例程中代码核心部分如下：

```java
// First move the robot to a suitable joint configuration
    std::array<double, 7> q_goal = {
            
   
     
     {
            
   
     
     0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
    MotionGenerator motion_generator(0.5, q_goal);
    std::cout << "WARNING: This example will move the robot! "
              << "Please make sure to have the user stop button at hand!" << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();
    robot.control(motion_generator);
    std::cout << "Finished moving to initial joint configuration." << std::endl;
    // Set additional parameters always before the control loop, NEVER in the control loop!
    // Set collision behavior.
    robot.setCollisionBehavior(
        {
            
   
     
     {
            
   
     
     20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {
            
   
     
     {
            
   
     
     20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
        {
            
   
     
     {
            
   
     
     20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {
            
   
     
     {
            
   
     
     20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
        {
            
   
     
     {
            
   
     
     20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {
            
   
     
     {
            
   
     
     20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
        {
            
   
     
     {
            
   
     
     20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {
            
   
     
     {
            
   
     
     20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});
    std::array<double, 16> initial_pose;
    double time = 0.0;
    robot.control([&time, &initial_pose](const franka::RobotState& robot_state,
                                         franka::Duration period) -> franka::CartesianPose {
            
   
     
     
      time += period.toSec();
      if (time == 0.0) {
            
   
     
     
        initial_pose = robot_state.O_T_EE_c;
      }
      constexpr double kRadius = 0.3;
      double angle = M_PI / 4 * (1 - std::cos(M_PI / 5.0 * time));
      double delta_x = kRadius * std::sin(angle);
      double delta_z = kRadius * (std::cos(angle) - 1);
      std::array<double, 16> new_pose = initial_pose;
      new_pose[12] += delta_x;
      new_pose[14] += delta_z;
      if (time >= 10.0) {
            
   
     
     
        std::cout << std::endl << "Finished motion, shutting down example" << std::endl;
        return franka::MotionFinished(new_pose);
      }
      return new_pose;
    });
```

这个程序控制机器人末端在y-z平面执行圆周运动。其原理十分简单，单纯是利用时间插值画了一个圆形。与关节空间轨迹生成代码唯一的区别是control函数输出变成了`franka::CartesianPose`。位姿数据，即 S E ( 3 ) SE(3) SE(3)数据是16维的，每4维代表  S E ( 3 ) SE(3) SE(3) 中的一列。注意本例子中直接采用一个16维的array模板类的对象作为输出返回，虽然可以这样但是为了防止出错，笔者还是推荐采用libfranka库中的`franka::CartesianPose`类。上述例子中机器人末端没有姿态的变化，通常不会出问题，只要做好插值不要有太大的阶跃输入。

速度轨迹的生成笔者并不常用，此处也不做介绍。读者请参考[官方例程][Link 11]。

## 阻抗控制与力控 

Franka机器人的一大优势就是直接实时控制关节力矩，这让用户可以自由设计复杂控制策略。

### 以力矩为输入的运动控制 

先上例程（仅核心部分）：

```java
// Set and initialize trajectory parameters.
  const double radius = 0.05;
  const double vel_max = 0.25;
  const double acceleration_time = 2.0;
  const double run_time = 20.0;

  double vel_current = 0.0;
  double angle = 0.0;
  double time = 0.0;
    // Define callback function to send Cartesian pose goals to get inverse kinematics solved.
    auto cartesian_pose_callback = [=, &time, &vel_current, &running, &angle, &initial_pose](
                                       const franka::RobotState& robot_state,
                                       franka::Duration period) -> franka::CartesianPose {
            
   
     
     
      time += period.toSec();
      if (time == 0.0) {
            
   
     
     
        // Read the initial pose to start the motion from in the first time step.
        initial_pose = robot_state.O_T_EE_c;
      }
      // Compute Cartesian velocity.
      if (vel_current < vel_max && time < run_time) {
            
   
     
     
        vel_current += period.toSec() * std::fabs(vel_max / acceleration_time);
      }
      if (vel_current > 0.0 && time > run_time) {
            
   
     
     
        vel_current -= period.toSec() * std::fabs(vel_max / acceleration_time);
      }
      vel_current = std::fmax(vel_current, 0.0);
      vel_current = std::fmin(vel_current, vel_max);
      // Compute new angle for our circular trajectory.
      angle += period.toSec() * vel_current / std::fabs(radius);
      if (angle > 2 * M_PI) {
            
   
     
     
        angle -= 2 * M_PI;
      }
      // Compute relative y and z positions of desired pose.
      double delta_y = radius * (1 - std::cos(angle));
      double delta_z = radius * std::sin(angle);
      franka::CartesianPose pose_desired = initial_pose;
      pose_desired.O_T_EE[13] += delta_y;
      pose_desired.O_T_EE[14] += delta_z;
      // Send desired pose.
      if (time >= run_time + acceleration_time) {
            
   
     
     
        running = false;
        return franka::MotionFinished(pose_desired);
      }
      return pose_desired;
    };
    // Set gains for the joint impedance control.
    // Stiffness
    const std::array<double, 7> k_gains = {
            
   
     
     {
            
   
     
     600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0}};
    // Damping
    const std::array<double, 7> d_gains = {
            
   
     
     {
            
   
     
     50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0}};
    // Define callback for the joint torque control loop.
    std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
        impedance_control_callback =
            [&print_data, &model, k_gains, d_gains](
                const franka::RobotState& state, franka::Duration /*period*/) -> franka::Torques {
            
   
     
     
      // Read current coriolis terms from model.
      std::array<double, 7> coriolis = model.coriolis(state);
      // Compute torque command from joint impedance control law.
      // Note: The answer to our Cartesian pose inverse kinematics is always in state.q_d with one
      // time step delay.
      std::array<double, 7> tau_d_calculated;
      for (size_t i = 0; i < 7; i++) {
            
   
     
     
        tau_d_calculated[i] =
            k_gains[i] * (state.q_d[i] - state.q[i]) - d_gains[i] * state.dq[i] + coriolis[i];
      }
      // Send torque command.
      return tau_d_rate_limited;
    };
    // Start real-time control loop.
    robot.control(impedance_control_callback, cartesian_pose_callback);
```

例程分析：

 *  这个程序完成了什么功能？——控制机器人末端在y-z平面执行圆周运动。
 *  这个程序与前述运动生成有何区别？——采用了专门定义的控制器，并非默认阻抗控制器。
 *  理论上是如何实现的？——梯形速度规划，原理比较简单，此处不做详述。
 *  值得注意的是：控制信号是关节力矩，控制目标是机器人末端笛卡尔空间的运动。那么这就绕不开机器人的逆向运动学问题（逆解）。然而Franka机器人是个7自由度冗余的机构，逆解问题是 ill-posed problem 。好在 Franka 机器人提供自带的逆解，即  q d q\_\{d\} qd ，注意有一个控制周期的延迟，也就是上一时刻的逆解。这个逆解虽然能用，但是插值做不好也容易出问题，建议慎用之。

理清上述问题后，再来看程序：  
首先，初始化控制参数，包括圆周半径、最大速度、加速时间、系统运行时间。

```java
const double radius = 0.05;
  const double vel_max = 0.25;
  const double acceleration_time = 2.0;
  const double run_time = 20.0;
```

后面的部分分成两块看，首先是运动规划部分。  
初始位姿是当前位姿。

```java
time += period.toSec();
      if (time == 0.0) {
            
   
     
     
        // Read the initial pose to start the motion from in the first time step.
        initial_pose = robot_state.O_T_EE_c;
      }
```

随后，进行梯形速度规划

```java
if (vel_current < vel_max && time < run_time) {
            
   
     
     
        vel_current += period.toSec() * std::fabs(vel_max / acceleration_time);
      }
      if (vel_current > 0.0 && time > run_time) {
            
   
     
     
        vel_current -= period.toSec() * std::fabs(vel_max / acceleration_time);
      }
      vel_current = std::fmax(vel_current, 0.0);
      vel_current = std::fmin(vel_current, vel_max);
```

根据线速度更新角度，注意角度不能超过  2 π 2\\pi 2π。

```java
// Compute new angle for our circular trajectory.
      angle += period.toSec() * vel_current / std::fabs(radius);
      if (angle > 2 * M_PI) {
            
   
     
     
        angle -= 2 * M_PI;
      }
```

更新笛卡尔空间位姿。

```java
// Compute relative y and z positions of desired pose.
      double delta_y = radius * (1 - std::cos(angle));
      double delta_z = radius * std::sin(angle);
      franka::CartesianPose pose_desired = initial_pose;
      pose_desired.O_T_EE[13] += delta_y;
      pose_desired.O_T_EE[14] += delta_z;
```

然后是力矩控制律。  
首先设定控制器参数，这里是刚度阻尼参数。

```java
// Set gains for the joint impedance control.
// Stiffness
const std::array<double, 7> k_gains = {
            
   
     
     {
            
   
     
     600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0}};
// Damping
const std::array<double, 7> d_gains = {
            
   
     
     {
            
   
     
     50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0}};
```

读取科氏力补偿，设置控制律： τ d = K ( q d − q ) − D q ˙ + C ( q , q ˙ ) \\tau\_\{d\} = K(q\_\{d\} - q) - D\\dot\{q\} + C(q,\\dot\{q\}) τd=K(qd−q)−Dq˙\+C(q,q˙)

```java
// Read current coriolis terms from model.
      std::array<double, 7> coriolis = model.coriolis(state);
      // Compute torque command from joint impedance control law.
      // Note: The answer to our Cartesian pose inverse kinematics is always in state.q_d with one
      // time step delay.
      std::array<double, 7> tau_d_calculated;
      for (size_t i = 0; i < 7; i++) {
            
   
     
     
        tau_d_calculated[i] =
            k_gains[i] * (state.q_d[i] - state.q[i]) - d_gains[i] * state.dq[i] + coriolis[i];
      }
      // Send torque command.
      return tau_d_rate_limited;
```

### 外力估计与控制 

假设读者对Eigen模板库有初步了解，先上例程（仅核心部分）：

```java
Eigen::VectorXd initial_tau_ext(7), tau_error_integral(7);
    // Bias torque sensor
    std::array<double, 7> gravity_array = model.gravity(initial_state);
    Eigen::Map<Eigen::Matrix<double, 7, 1>> initial_tau_measured(initial_state.tau_J.data());
    Eigen::Map<Eigen::Matrix<double, 7, 1>> initial_gravity(gravity_array.data());
    initial_tau_ext = initial_tau_measured - initial_gravity;
    // init integrator
    tau_error_integral.setZero();
    // define callback for the torque control loop
    Eigen::Vector3d initial_position;
    double time = 0.0;
    auto get_position = [](const franka::RobotState& robot_state) {
            
   
     
     
      return Eigen::Vector3d(robot_state.O_T_EE[12], robot_state.O_T_EE[13],
                             robot_state.O_T_EE[14]);
    };
    auto force_control_callback = [&](const franka::RobotState& robot_state,
                                      franka::Duration period) -> franka::Torques {
            
   
     
     
      time += period.toSec();
      if (time == 0.0) {
            
   
     
     
        initial_position = get_position(robot_state);
      }
      if (time > 0 && (get_position(robot_state) - initial_position).norm() > 0.01) {
            
   
     
     
        throw std::runtime_error("Aborting; too far away from starting pose!");
      }
      // get state variables
      std::array<double, 42> jacobian_array =
          model.zeroJacobian(franka::Frame::kEndEffector, robot_state);
      Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
      Eigen::Map<const Eigen::Matrix<double, 7, 1>> tau_measured(robot_state.tau_J.data());
      Eigen::Map<const Eigen::Matrix<double, 7, 1>> gravity(gravity_array.data());
      Eigen::VectorXd tau_d(7), desired_force_torque(6), tau_cmd(7), tau_ext(7);
      desired_force_torque.setZero();
      desired_force_torque(2) = desired_mass * -9.81;
      tau_ext << tau_measured - gravity - initial_tau_ext;
      tau_d << jacobian.transpose() * desired_force_torque;
      tau_error_integral += period.toSec() * (tau_d - tau_ext);
      // FF + PI control
      tau_cmd << tau_d + k_p * (tau_d - tau_ext) + k_i * tau_error_integral;
      // Smoothly update the mass to reach the desired target value
      desired_mass = filter_gain * target_mass + (1 - filter_gain) * desired_mass;
      std::array<double, 7> tau_d_array{
            
   
     
     };
      Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_cmd;
      return tau_d_array;
    };
```

程序乍一看有些复杂，我们拆分来看。

 *  首先，这个程序用以实现什么功能?——假设机器人末端与一个刚性平面垂直接触，控制机器人末端z轴方向产生一个1kg的力（ 9.81 × 1 ( N ) 9.81\\times 1(N) 9.81×1(N)）。
 *  理论上是如何实现的？——以关节力矩作为输入的PI控制器： τ c = τ d + K P ( τ d − τ e x t ) + K I ∫ 0 t τ e ( s ) d s \\tau\_\{c\} = \\tau\_\{d\} + K\_\{P\}(\\tau\_\{d\} - \\tau\_\{ext\}) + K\_\{I\}\\int\_\{0\}^\{t\}\\tau\_\{e\}(s)\\mathrm\{d\}s τc=τd\+KP(τd−τext)\+KI∫0tτe(s)ds 其中  τ c \\tau\_\{c\} τc 是力矩指令，  τ d \\tau\_\{d\} τd 是期望力矩， τ e x t \\tau\_\{ext\} τext 是外部力矩，  τ e \\tau\_\{e\} τe 是偏差。 τ e = τ d − τ e x t \\tau\_\{e\} =\\tau\_\{d\} -\\tau\_\{ext\} τe=τd−τext。
 *  这里多说一句 τ e x t \\tau\_\{ext\} τext：这个值并非测量值  τ m \\tau\_\{m\} τm，而是  τ e x t ( t ) = τ m ( t ) − g ( q ) − τ e x t ( 0 ) \\tau\_\{ext\}(t) = \\tau\_\{m\}(t) - g(q) -\\tau\_\{ext\}(0) τext(t)=τm(t)−g(q)−τext(0)。也就是代码注释中所提及的 bias torque sensor。也就是说，通过RobotStarte结构体读取的力矩（tau\_J）本身并不包含重力补偿，是单纯的传感器测量值。同时，使用时还要注意初始状态下必然存在偏置力矩（ τ e x t ( 0 ) \\tau\_\{ext\}(0) τext(0)）。
 *  本实例是在笛卡尔空间控制力，然而控制信号是关节力矩，因此需要通过雅可比矩阵推算期望力矩  τ d \\tau\_\{d\} τd。好在libfranka的模型库 franka::Model 提供了雅可比矩阵的计算方法，可以直接调用： τ d = J T ( q ) f d \\tau\_\{d\} = J^\{T\}(q)f\_\{d\} τd=JT(q)fd。

理清上述问题之后，再看程序就容易许多。  
初始化偏置  τ e x t ( 0 ) \\tau\_\{ext\}(0) τext(0)：

```java
// Bias torque sensor
std::array<double, 7> gravity_array = model.gravity(initial_state);
Eigen::Map<Eigen::Matrix<double, 7, 1>> initial_tau_measured(initial_state.tau_J.data());
Eigen::Map<Eigen::Matrix<double, 7, 1>> initial_gravity(gravity_array.data());
initial_tau_ext = initial_tau_measured - initial_gravity;
```

保险起见，确保机器人在执行程序过程中不产生较大位移：

```java
if (time > 0 && (get_position(robot_state) - initial_position).norm() > 0.01) {
            
   
     
     
	throw std::runtime_error("Aborting; too far away from starting pose!");
}
```

读取反馈信号：雅可比矩阵（Spatial Jacobian）、力矩测量值、重力：

```java
// get state variables
std::array<double, 42> jacobian_array =
          model.zeroJacobian(franka::Frame::kEndEffector, robot_state);
Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
Eigen::Map<const Eigen::Matrix<double, 7, 1>> tau_measured(robot_state.tau_J.data());
Eigen::Map<const Eigen::Matrix<double, 7, 1>> gravity(gravity_array.data());
```

计算期望力矩  τ d \\tau\_\{d\} τd：

```java
desired_force_torque.setZero();
desired_force_torque(2) = desired_mass * -9.81;
tau_ext << tau_measured - gravity - initial_tau_ext;
tau_d << jacobian.transpose() * desired_force_torque;
```

误差积分，PI控制：

```java
tau_error_integral += period.toSec() * (tau_d - tau_ext);
// FF + PI control
tau_cmd << tau_d + k_p * (tau_d - tau_ext) + k_i * tau_error_integral;
// Smoothly update the mass to reach the desired target value
desired_mass = filter_gain * target_mass + (1 - filter_gain) * desired_mass;
```

注意最后一行，这是一个控制目标平滑过渡的过程： m d = 0.001 m t + ( 1 − 0.001 ) m d m\_\{d\} = 0.001m\_\{t\} + (1-0.001)m\_\{d\} md=0.001mt\+(1−0.001)md， m d m\_\{d\} md的值终将趋于  m t m\_\{t\} mt。如前所述，给定阶跃控制目标会造成不友好的反复震荡。

### 阻抗控制 

阻抗控制因其简单性而被广泛使用，阻抗控制的基本方法本文不做介绍。先上笛卡尔空间阻抗控制官方例程（核心部分）：

```java
// Compliance parameters
  const double translational_stiffness{
            
   
     
     150.0};
  const double rotational_stiffness{
            
   
     
     10.0};
  Eigen::MatrixXd stiffness(6, 6), damping(6, 6);
  stiffness.setZero();
  stiffness.topLeftCorner(3, 3) << translational_stiffness * Eigen::MatrixXd::Identity(3, 3);
  stiffness.bottomRightCorner(3, 3) << rotational_stiffness * Eigen::MatrixXd::Identity(3, 3);
  damping.setZero();
  damping.topLeftCorner(3, 3) << 2.0 * sqrt(translational_stiffness) *
                                     Eigen::MatrixXd::Identity(3, 3);
  damping.bottomRightCorner(3, 3) << 2.0 * sqrt(rotational_stiffness) *
                                         Eigen::MatrixXd::Identity(3, 3);
    // connect to robot
    franka::Robot robot(argv[1]);
    setDefaultBehavior(robot);
    // load the kinematics and dynamics model
    franka::Model model = robot.loadModel();
    franka::RobotState initial_state = robot.readOnce();
    // equilibrium point is the initial position
    Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));
    Eigen::Vector3d position_d(initial_transform.translation());
    Eigen::Quaterniond orientation_d(initial_transform.linear());
    // set collision behavior
    robot.setCollisionBehavior({
            
   
     
     {
            
   
     
     100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                               {
            
   
     
     {
            
   
     
     100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                               {
            
   
     
     {
            
   
     
     100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                               {
            
   
     
     {
            
   
     
     100.0, 100.0, 100.0, 100.0, 100.0, 100.0}});
    // define callback for the torque control loop
    std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
        impedance_control_callback = [&](const franka::RobotState& robot_state,
                                         franka::Duration /*duration*/) -> franka::Torques {
            
   
     
     
      // get state variables
      std::array<double, 7> coriolis_array = model.coriolis(robot_state);
      std::array<double, 42> jacobian_array =
          model.zeroJacobian(franka::Frame::kEndEffector, robot_state);
      // convert to Eigen
      Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
      Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
      Eigen::Map<const Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
      Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
      Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
      Eigen::Vector3d position(transform.translation());
      Eigen::Quaterniond orientation(transform.linear());
      // compute error to desired equilibrium pose
      // position error
      Eigen::Matrix<double, 6, 1> error;
      error.head(3) << position - position_d;
      // orientation error
      // "difference" quaternion
      if (orientation_d.coeffs().dot(orientation.coeffs()) < 0.0) {
            
   
     
     
        orientation.coeffs() << -orientation.coeffs();
      }
      // "difference" quaternion
      Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d);
      error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
      // Transform to base frame
      error.tail(3) << -transform.linear() * error.tail(3);
      // compute control
      Eigen::VectorXd tau_task(7), tau_d(7);
      // Spring damper system with damping ratio=1
      tau_task << jacobian.transpose() * (-stiffness * error - damping * (jacobian * dq));
      tau_d << tau_task + coriolis;

      std::array<double, 7> tau_d_array{
            
   
     
     };
      Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_d;
      return tau_d_array;
    };
```

我们依然拆分开解析：

 *  这个程序完成了什么功能?——固定末端位姿的阻抗控制，即机器人末端模拟一个弹簧阻尼机构。
 *  理论上是如何实现的？ τ d = J T ( q ) ( − α ⋅ e − β ⋅ J ( q ) q ˙ ) + C ( q , q ˙ ) \\tau\_\{d\} = J^\{T\}(q)\\left( -\\alpha\\cdot e - \\beta\\cdot J(q)\\dot\{q\} \\right) + C(q,\\dot\{q\}) τd=JT(q)(−α⋅e−β⋅J(q)q˙)\+C(q,q˙) 其中  α \\alpha α、 β \\beta β 分别为刚度（弹簧）和阻尼， e e e 是位姿偏差。科氏力  C ( q , q ˙ ) C(q,\\dot\{q\}) C(q,q˙) 比较小，影响不大。注意机器人本身自带重力补偿。
 *  本例中雅可比矩阵和科氏力均是通过 libfranka 提供的 `franka::Model` 类获取，注意构造函数输入可以是 `franka::RobotState` 结构体。
 *  这里多说一点偏差  e e e ，这个偏差时位姿与姿态的混合偏差 （ e ∈ R 6 e\\in\\mathbb\{R\}^\{6\} e∈R6），此处位置偏差用欧式距离，姿态偏差用四元数虚部之差。这个做法在工程上可用，但是并非十分完善，该问题此处暂不做讨论。

理清上述问题后，再看程序就没有那么复杂。  
首先定义刚度阻尼参数：

```java
const double translational_stiffness{
            
   
     
     150.0};
  const double rotational_stiffness{
            
   
     
     10.0};
  Eigen::MatrixXd stiffness(6, 6), damping(6, 6);
  stiffness.setZero();
  stiffness.topLeftCorner(3, 3) << translational_stiffness * Eigen::MatrixXd::Identity(3, 3);
  stiffness.bottomRightCorner(3, 3) << rotational_stiffness * Eigen::MatrixXd::Identity(3, 3);
  damping.setZero();
  damping.topLeftCorner(3, 3) << 2.0 * sqrt(translational_stiffness) *
                                     Eigen::MatrixXd::Identity(3, 3);
  damping.bottomRightCorner(3, 3) << 2.0 * sqrt(rotational_stiffness) *
                                         Eigen::MatrixXd::Identity(3, 3);
```

此处刚度  α ∈ R 6 × 6 \\alpha \\in \\mathbb\{R\}^\{6\\times 6\} α∈R6×6，阻尼  β ∈ R 6 × 6 \\beta\\in\\mathbb\{R\}^\{6\\times 6\} β∈R6×6，且左上角  3 × 3 3\\times 3 3×3 区块为位置刚度/阻尼，右下角  3 × 3 3\\times 3 3×3 区块为姿态刚度/阻尼。这里设定每个维度上阻尼相同。  
然后，设置平衡点为当前位姿：

```java
Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));
    Eigen::Vector3d position_d(initial_transform.translation());
    Eigen::Quaterniond orientation_d(initial_transform.linear());
```

注意姿态由四元数表示。更准确地说应该是单位四元数。  
控制策略开始后，实时读取当前状态，即反馈信号：位置、速度、末端姿态、当前雅可比矩阵、当前科氏力。

```java
// get state variables
      std::array<double, 7> coriolis_array = model.coriolis(robot_state);
      std::array<double, 42> jacobian_array =
          model.zeroJacobian(franka::Frame::kEndEffector, robot_state);
      // convert to Eigen
      Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
      Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
      Eigen::Map<const Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
      Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
      Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
      Eigen::Vector3d position(transform.translation());
      Eigen::Quaterniond orientation(transform.linear());
```

注意通过 libfranka 读取的数据都是 `std::array` 模板类的对象，我们需要将其转化成 `Eigen::Matrix` 模板类的对象。  
下一步，计算偏差：

```java
// position error
      Eigen::Matrix<double, 6, 1> error;
      error.head(3) << position - position_d;
      // orientation error
      // "difference" quaternion
      if (orientation_d.coeffs().dot(orientation.coeffs()) < 0.0) {
            
   
     
     
        orientation.coeffs() << -orientation.coeffs();
      }
      Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d);
      error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
      // Transform to base frame
      error.tail(3) << -transform.linear() * error.tail(3);
```

位置偏差采用欧式距离，直接用当前末端位置减平衡点位置即可。姿态偏差采用四元数减法。这里稍微详细地说一下：由于同一个姿态可以用两个单位四元数表示，即  Q Q Q 与  − Q -Q −Q 表达的是同一个姿态，所以在计算差之前首先调整符号；另外，此处采用四元数虚部来表示距离，这个定义在小范围的偏差上可用，更完备的黎曼度量等问题此处不探讨；最后，需要注意坐标系的变换。  
最后，计算控制律：

```java
tau_task << jacobian.transpose() * (-stiffness * error - damping * (jacobian * dq));
      tau_d << tau_task + coriolis;
```

注意返回值的类型不能是 `Eigen::Matrix` ，需要变回 `std::array` 。

## 其它参考 

 *  [libfranka官方：github.io][libfranka]
 *  [libfranka官方：FCI document][Franka]
 *  [Eigen主页][Eigen]
 *  [非官方：Eigen库使用指南][Eigen 1]
 *  [非官方：Eigen与VS Code配置][Eigen_VS Code]
 *  [非官方：Eigen快速入门][Eigen 2]
 *  [非官方：auto关键字的用法][auto]


[Franka]: https://frankaemika.github.io/docs/
[libfranka]: https://frankaemika.github.io/libfranka/
[libfranka 1]: #libfranka_13
[Link 1]: #_22
[Link 2]: #_26
[Link 3]: #_77
[Link 4]: #_122
[Link 5]: #_124
[Link 6]: #_280
[Link 7]: #_376
[Link 8]: #_521
[Link 9]: https://github.com/frankaemika/libfranka
[Link 10]: https://blog.csdn.net/philthinker/article/details/106340862
[FCI]: https://frankaemika.github.io/docs/control_parameters.html
[FCI 1]: https://frankaemika.github.io/docs/libfranka.html
[frankaemika.github.io]: https://frankaemika.github.io/libfranka/classfranka_1_1Robot.html#a2da598c539469827409ac7e3bb61d5da
[Link 11]: https://frankaemika.github.io/libfranka/examples.html
[Eigen]: http://eigen.tuxfamily.org/index.php?title=Main_Page
[Eigen 1]: https://www.jianshu.com/p/931dff3b1b21
[Eigen_VS Code]: https://www.geek-share.com/detail/2769244138.html
[Eigen 2]: https://www.cnblogs.com/python27/p/EigenQuickRef.html
[auto]: https://www.cnblogs.com/QG-whz/p/4951177.html