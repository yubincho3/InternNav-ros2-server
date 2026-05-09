from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    rgb_topic_arg     = DeclareLaunchArgument('rgb_topic',     description='RGB image topic name')
    s1_model_path_arg = DeclareLaunchArgument('s1_model_path', description='System1 model path')
    s2_model_path_arg = DeclareLaunchArgument('s2_model_path', description='System2 model path')
    s1_device_arg     = DeclareLaunchArgument('s1_device',     default_value='cuda:0', description='System1 CUDA device')
    s2_device_arg     = DeclareLaunchArgument('s2_device',     default_value='cuda:1', description='System2 CUDA device')

    server_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('internnav_server'), 'launch', 'realworld.launch.py'
            ])
        ]),
        launch_arguments={
            'rgb_topic':     LaunchConfiguration('rgb_topic'),
            's1_model_path': LaunchConfiguration('s1_model_path'),
            's2_model_path': LaunchConfiguration('s2_model_path'),
            's1_device':     LaunchConfiguration('s1_device'),
            's2_device':     LaunchConfiguration('s2_device'),
        }.items()
    )

    manager_node = Node(
        package='internnav_manager',
        executable='manager',
        name='internnav_manager',
        output='screen',
    )

    return LaunchDescription([
        rgb_topic_arg,
        s1_model_path_arg,
        s2_model_path_arg,
        s1_device_arg,
        s2_device_arg,
        server_launch,
        manager_node,
    ])
