import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, EnvironmentVariable
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _find_workspace_root(start: Path) -> Path:
    """向上查找包含 planner_lib 的目录，找不到则返回 start 的父目录。"""
    cur = start
    for _ in range(8):  # 足够覆盖源码或 install 两种路径
        if (cur / 'planner_lib').exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.parent


def generate_launch_description():
    # 运行时需要 planner_lib 和 gtsam 提供的 .so，使用 launch 自动补 LD_LIBRARY_PATH，
    # 避免遗漏导致“exit code 127: libmap_manager.so not found”。
    file_path = Path(__file__).resolve()
    root_dir = _find_workspace_root(file_path)
    planner_lib_dir = root_dir / 'planner_lib'
    gtsam_lib_dir = planner_lib_dir / '3rdparty' / 'gtsam-4.1.1' / 'install' / 'lib'

    pkg_share = FindPackageShare('pct_planner_cpp_port')
    default_tomo = PathJoinSubstitution([pkg_share, 'rsc', 'tomogram', 'scene_map.bin'])
    default_pcd = PathJoinSubstitution([pkg_share, 'trajectory.pcd'])
    default_params = PathJoinSubstitution([pkg_share, 'config', 'scene_default.yaml'])

    output_path = LaunchConfiguration('output_path')
    pcd_path = LaunchConfiguration('pcd_path')
    tomo_path = LaunchConfiguration('tomo_path')
    params_file = LaunchConfiguration('params_file')
    enable_planner = LaunchConfiguration('enable_planner')
    enable_planner_direct = LaunchConfiguration('enable_planner_direct')
    enable_pcd_publisher = LaunchConfiguration('enable_pcd_publisher')
    publish_start_end = LaunchConfiguration('publish_start_end')
    start_x = LaunchConfiguration('start_x')
    start_y = LaunchConfiguration('start_y')
    start_z = LaunchConfiguration('start_z')
    end_x = LaunchConfiguration('end_x')
    end_y = LaunchConfiguration('end_y')
    end_z = LaunchConfiguration('end_z')

    return LaunchDescription([
        # 确保运行时能找到 planner_lib 与 gtsam 动态库
        SetEnvironmentVariable(
            'LD_LIBRARY_PATH',
            value=[
                str(planner_lib_dir), ':', str(gtsam_lib_dir), ':',
                EnvironmentVariable('LD_LIBRARY_PATH')
            ]
        ),

        # Paths and core params
        DeclareLaunchArgument('output_path', default_value=default_tomo,
                              description='tomography_node 输出 tomogram 的路径'),
        DeclareLaunchArgument('pcd_path', default_value=default_pcd,
                              description='planner_* 输出 ASCII PCD 的路径'),
        DeclareLaunchArgument('tomo_path', default_value=default_tomo,
                              description='planner_direct_node 读取 tomogram 的路径'),
        DeclareLaunchArgument('params_file', default_value=default_params,
                  description='统一参数文件 (YAML)，覆盖所有节点'),

        # Optional components
        DeclareLaunchArgument('enable_planner', default_value='true',
                              description='是否启动 planner_node（在线 tomogram）'),
        DeclareLaunchArgument('enable_planner_direct', default_value='false',
                              description='是否启动 planner_direct_node（离线 tomogram 文件）'),
        DeclareLaunchArgument('enable_pcd_publisher', default_value='true',
                              description='是否启动示例点云发布器 /global_points'),
        DeclareLaunchArgument('publish_start_end', default_value='false',
                              description='是否自动发布一次 /start_pos /end_pos'),
        DeclareLaunchArgument('start_x', default_value='0.0'),
        DeclareLaunchArgument('start_y', default_value='0.0'),
        DeclareLaunchArgument('start_z', default_value='0.0'),
        DeclareLaunchArgument('end_x', default_value='5.0'),
        DeclareLaunchArgument('end_y', default_value='5.0'),
        DeclareLaunchArgument('end_z', default_value='0.0'),
        
        # Tomogram construction
        Node(
            package='pct_planner_cpp_port',
            executable='tomography_node',
            name='pct_tomography_cpp',
            output='screen',
            parameters=[params_file],
        ),

        # Online planner (consumes /tomogram_data)
        Node(
            condition=IfCondition(enable_planner),
            package='pct_planner_cpp_port',
            executable='planner_node',
            name='pct_planner_cpp',
            output='screen',
            parameters=[params_file],
        ),

        # Offline planner (loads tomogram file)
        # 注意：命令行参数放在 params_file 之后，确保优先级更高
        Node(
            condition=IfCondition(enable_planner_direct),
            package='pct_planner_cpp_port',
            executable='planner_direct_node',
            name='pct_planner_direct_cpp',
            output='screen',
            parameters=[
                params_file,
                {'tomo_path': tomo_path},
                {'pcd_path': pcd_path},
            ],
        ),

        # Sample point cloud publisher
        Node(
            condition=IfCondition(enable_pcd_publisher),
            package='pct_planner_cpp_port',
            executable='pcd_publisher',
            name='pct_pcd_publisher',
            output='screen',
        ),

        # Optional one-shot start/end publishers (delayed to allow nodes ready)
        TimerAction(
            condition=IfCondition(publish_start_end),
            period=5.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        'ros2', 'topic', 'pub', '--once', '/start_pos',
                        'geometry_msgs/msg/Point',
                        ['{x: ', start_x, ', y: ', start_y, ', z: ', start_z, '}']
                    ],
                    output='screen',
                ),
                ExecuteProcess(
                    cmd=[
                        'ros2', 'topic', 'pub', '--once', '/end_pos',
                        'geometry_msgs/msg/Point',
                        ['{x: ', end_x, ', y: ', end_y, ', z: ', end_z, '}']
                    ],
                    output='screen',
                ),
            ],
        ),
    ])
