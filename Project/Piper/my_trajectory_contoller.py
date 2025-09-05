#!/usr/bin/env python3
# -*-coding:utf8-*-
'''
@author: Salvador 万鹏
@time: 2025-09-04
@note: 本文件调用TrajectoryController中execute_trajectory方法，输入离散点完成轨迹控制
需要额外注意的是，可以通过数采/控制的频率来实现速度的控制，而不是修改速度参数。
数采的频率就算再大也无所谓，因为最后的结果是退化到100hz控制效果
我已经尝试了闭环控制，这是一个不现实的方法
轨迹生成器或者接受器请在generate_trajectory()方法中定义
'''
import time
from piper_sdk import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

if_plt = True # 默认开启绘图模式，确保安全，也可以关掉。
speed = 35  # 速度调整接口，可以在这里面修改运行速度，可合理调整，但是不建议修改
time_interval = 0.02 # 0.01s对应100Hz的控制频率，如果想降低运行速度（或者适应数采设的频率）可以加大time_interval
# ！！！！！！！！！！！！！注意！！！！！！
# 最大的控制频率就是100Hz！
orientation_trajectory_flag = 0
position_trajectory_flag = 0
class TrajectoryVisualizer:
    """轨迹可视化工具类"""
    
    @staticmethod
    def animate_flange_orientation(points, interval=50):
        """
        特点：
        1. 坐标系永远锁定在视图中心
        2. 拖动时不会跑偏
        3. 坐标轴标识永久可见
        4. 法兰箭头强化显示
        """
        points = np.array(points)
        pos = points[:, :3]
        euler_angles = points[:, 3:6]

        # 创建图形和3D坐标轴
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 计算数据范围
        max_range = np.max(np.abs(pos)) * 1.8  # 扩大20%的显示范围
        axis_length = max_range * 1.5  # 坐标轴长度略大于数据范围

        # ================= 固定坐标系绘制 =================
        # 主坐标轴（加粗+永久显示）
        ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', linewidth=4, 
                arrow_length_ratio=0.15, alpha=0.7)
        ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', linewidth=4,
                arrow_length_ratio=0.15, alpha=0.7)
        ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', linewidth=4,
                arrow_length_ratio=0.15, alpha=0.7)
        
        # 坐标轴标签（永久固定位置）
        label_offset = axis_length * 0.1  # 标签偏移量
        ax.text(axis_length+label_offset, 0, 0, 'X∞', color='r', 
                fontsize=14, fontweight='bold')
        ax.text(0, axis_length+label_offset, 0, 'Y∞', color='g', 
                fontsize=14, fontweight='bold')
        ax.text(0, 0, axis_length+label_offset, 'Z∞', color='b', 
                fontsize=14, fontweight='bold')

        # ================= 动态元素初始化 =================
        # 局部坐标系箭头（缩短显示）
        frame_length = max_range * 0.25
        x_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='darkred', linewidth=3)
        y_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='darkgreen', linewidth=3)
        z_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='darkblue', linewidth=3)
        
        # 法兰箭头（工业级醒目设计）
        flange_arrow = ax.quiver(0, 0, 0, 0, 0, 0,
                            color='#FFA500',  # 橙色更醒目
                            linewidth=6,
                            arrow_length_ratio=0.25,
                            alpha=0.95)

        # 轨迹线
        trajectory_line, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.6)
        current_point = ax.scatter([], [], [], c='r', s=150, alpha=0.9)

        # ================= 视图锁定配置 =================
        def set_axes_limits():
            """锁定视图范围，防止拖动跑偏"""
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])
            ax.set_box_aspect([1,1,1])

        set_axes_limits()
        
        # 响应视图变化事件
        def on_move(event):
            if event.inaxes == ax:
                set_axes_limits()
        
        fig.canvas.mpl_connect('motion_notify_event', on_move)

        # ================= 动画更新 =================
        def update(frame):
            current_pos = pos[frame]
            euler = euler_angles[frame]
            
            rotation = R.from_euler('xyz', euler, degrees=True)
            rot_mat = rotation.as_matrix()
            
            # 更新局部坐标系
            x_dir = rot_mat[:, 0] * frame_length
            y_dir = rot_mat[:, 1] * frame_length
            z_dir = rot_mat[:, 2] * frame_length
            
            x_arrow.set_segments([[current_pos, current_pos + x_dir]])
            y_arrow.set_segments([[current_pos, current_pos + y_dir]])
            z_arrow.set_segments([[current_pos, current_pos + z_dir]])
            
            # 更新法兰箭头（1.8倍长度突出显示）
            flange_dir = z_dir * 1.8
            flange_arrow.set_segments([[current_pos, current_pos + flange_dir]])
            
            # 更新轨迹
            trajectory_line.set_data(pos[:frame+1, 0], pos[:frame+1, 1])
            trajectory_line.set_3d_properties(pos[:frame+1, 2])
            
            current_point._offsets3d = ([current_pos[0]], [current_pos[1]], [current_pos[2]])
            
            ax.set_title(
                f'Frame: {frame+1}/{len(points)}\n'
                f'Position: [{current_pos[0]:.1f}, {current_pos[1]:.1f}, {current_pos[2]:.1f}]\n'
                f'Rotation: RX={euler[0]:.1f}° RY={euler[1]:.1f}° RZ={euler[2]:.1f}°',
                fontsize=12, pad=15
            )
            
            return x_arrow, y_arrow, z_arrow, flange_arrow, trajectory_line, current_point

        # ================= 运行动画 =================
        try:
            ani = FuncAnimation(
                fig, update, frames=len(points),
                interval=interval, blit=False, repeat=True
            )
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"动画错误: {str(e)}")
            print("尝试更换后端: matplotlib.use('TkAgg')")
        
        return ani

    @staticmethod
    def plot_orientation_changes(points):
        """
        绘制末端执行器三轴姿态角变化曲线
        参数：
            points - 轨迹点数组，形状为(N,7)，包含[X,Y,Z,RX,RY,RZ,gripper]
        """
        points = np.array(points)
        rx = points[:, 3]  # RX角度（度）
        ry = points[:, 4]  # RY角度（度）
        rz = points[:, 5]  # RZ角度（度）
        
        # 创建带3个子图的画布
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        # 1. RX角度变化曲线（红色）
        ax1.plot(rx, 'r-', linewidth=1.5, label='RX')
        ax1.set_ylabel('RX (deg)', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(loc='upper right')
        
        # 2. RY角度变化曲线（绿色）
        ax2.plot(ry, 'g-', linewidth=1, label='RY')
        ax2.set_ylabel('RY (deg)', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc='upper right')
        
        # 3. RZ角度变化曲线（蓝色）
        ax3.plot(rz, 'b-', linewidth=1.5, label='RZ')
        ax3.set_ylabel('RZ (deg)', fontsize=10)
        ax3.set_xlabel('Trajectory Point Index', fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.legend(loc='upper right')
        
        # 添加总标题和调整布局
        plt.suptitle('End-Effector Orientation Changes During Trajectory', y=0.98, fontsize=12)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_infinite_3d(points, axis_length=1000):
        """
        模拟无限大3D空间的绘图函数
        - 坐标轴从原点(0,0,0)无限延伸（通过极大值模拟）
        - 自动聚焦到数据区域
        """
        points = np.array(points)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 1. 绘制轨迹数据
        ax.plot(points[:,0], points[:,1], points[:,2], 'b-', linewidth=2)
        ax.scatter(points[:,0], points[:,1], points[:,2], c='b', s=5)
        
        # 2. 绘制"无限大"坐标轴（用极大值模拟）
        ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', linewidth=3, arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', linewidth=3, arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0,axis_length, color='b', linewidth=3, arrow_length_ratio=0.1)
        
        # 3. 设置动态视图范围
        data_radius = np.max(np.abs(points)) * 1.5
        ax.set_xlim(-data_radius, data_radius)
        ax.set_ylim(-data_radius, data_radius)
        ax.set_zlim(-data_radius, data_radius)
        
        # 4. 添加永久坐标轴标签
        ax.text(axis_length, 0, 0, ' X∞', color='r', fontsize=12)
        ax.text(0, axis_length, 0, ' Y∞', color='g', fontsize=12)
        ax.text(0, 0, axis_length, ' Z∞', color='b', fontsize=12)
        
        ax.set_box_aspect([1,1,1])
        ax.set_xlabel('X → ∞', fontsize=12)
        ax.set_ylabel('Y → ∞', fontsize=12)
        ax.set_zlabel('Z → ∞', fontsize=12)
        plt.title('Infinite 3D Space Simulation', fontsize=14)
        plt.tight_layout()
        plt.show()


class TrajectoryController:
    """轨迹控制器类"""
    
    def __init__(self, can_port="can0"):
        self.piper = C_PiperInterface_V2(can_port)
        self.piper.ConnectPort()
        while not self.piper.EnablePiper():
            time.sleep(0.01)
        self.factor = 1000  # piper官方自定义的系数，不要动！
        self.trajectory = []  # 存储轨迹点
        self.current_index = 0
    
    def load_trajectory(self, points):
        """加载轨迹点列表，每个点应为[X,Y,Z,RX,RY,RZ]格式"""
        self.trajectory = points
        self.current_index = 0
    
    def execute_trajectory(self, speed, gripper_pos=0, gripper_speed=1000):
        """
        执行加载的轨迹
        :param speed: 运动速度 (0-100)[松灵的参数指定]
        :param gripper_pos: 夹爪位置
        :param gripper_speed: 夹爪速度
        """
        if len(self.trajectory) == 0:
            print("No trajectory loaded!")
            return
        
        print("Starting trajectory execution...")
        
        while self.current_index < len(self.trajectory):
            point = self.trajectory[self.current_index]
        
            # 打印当前末端位置（可选）
            X = round(point[0] * self.factor)
            Y = round(point[1] * self.factor)
            Z = round(point[2] * self.factor)
            RX = round(point[3] * self.factor)
            RY = round(point[4] * self.factor)
            RZ = round(point[5] * self.factor)
            
            # 控制机械臂运动
            self.piper.MotionCtrl_2(0x01, 0x00, speed, 0x00)
            self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
            
            # 控制夹爪（如果需要）
            if len(point) > 6:
                gripper_value = int(abs(float(point[6])))  # 确保转换为Python原生int
                self.piper.GripperCtrl(gripper_value, gripper_speed, 0x01, 0)
            else:
                self.piper.GripperCtrl(gripper_pos, gripper_speed, 0x01, 0)
            
            self.current_index += 1
            time.sleep(time_interval)  # 控制循环频率
        
        print("Trajectory execution completed!")


class TrajectoryGenerator:
    """轨迹生成器类"""
    
    @staticmethod
    def generate_orientation_trajectory():
        num_points = 500
        position = [250, 0, 350]
        global orientation_trajectory_flag
        orientation_trajectory_flag = 1
        """
        生成高频姿态变化轨迹
        参数：
            num_points - 轨迹点数
            position - 固定位置坐标[X,Y,Z]
        """
        # 生成高频变化的姿态角（单位：度）
        t = np.linspace(0, 4*np.pi, num_points)

        # 1. RX: 小幅正弦变化 (-15°~15°)
        rx_vals = 15 * np.sin(2*t)

        # 2. RY: 固定值90°
        ry_vals = np.full(num_points, 90.0)

        # 3. RZ: 限制在-30°~30°范围内摆动
        rz_vals = 30 * np.sin(t)

        # 构建轨迹点 [X,Y,Z,RX,RY,RZ,gripper]
        trajectory = np.column_stack([
            np.full(num_points, position[0]),  # X固定
            np.full(num_points, position[1]),  # Y固定  
            np.full(num_points, position[2]),  # Z固定
            rx_vals,  # RX小幅摆动
            ry_vals,  # RY固定
            rz_vals,  # RZ限制范围
            np.zeros(num_points)  # 夹爪关闭
        ])
        
        return trajectory
    
    @staticmethod
    def generate_trajectory():
        
        center=[250, 0, 350]
        radius=50
        num_points=250
        rx_val=20.0
        ry_val=120.0
        rz_val=0
        gripper_val=0
        global position_trajectory_flag
        position_trajectory_flag = 1
        """
        生成圆形轨迹
        参数：
            center - 圆心坐标[x,y,z]
            radius - 半径
            num_points - 点数
            rx_val, ry_val, rz_val - 固定姿态角
            gripper_val - 夹爪位置
        """
        theta = np.linspace(0, 2*np.pi, num_points)
        trajectory = np.array([
            [
                center[0] + radius * np.cos(t),  # X坐标    
                center[1] + radius * np.sin(t),  # Y坐标
                center[2],                       # Z坐标
                rx_val,                          # rx (固定值)
                ry_val,                          # ry (固定值)
                rz_val,                          # rz (固定值)
                np.int32(gripper_val)            # 夹爪位置                            
            ]
            for t in theta
        ])
        
        return trajectory


if __name__ == "__main__":
    # 创建控制器
    controller = TrajectoryController()
    
    # 生成高频姿态轨迹
    trajectory_generator = TrajectoryGenerator()
    position = trajectory_generator.generate_trajectory()

     # 可视化
    visualizer = TrajectoryVisualizer()
    if if_plt:
        if orientation_trajectory_flag == 1:
            visualizer.plot_orientation_changes(position)
            visualizer.animate_flange_orientation(position, interval=50)
            orientation_trajectory_flag = 0
        elif position_trajectory_flag == 1:
            visualizer.plot_infinite_3d(position)
            visualizer.animate_flange_orientation(position, interval=50)
            position_trajectory_flag = 0
    
    # 执行轨迹
    controller.load_trajectory(position)
    controller.execute_trajectory(speed)