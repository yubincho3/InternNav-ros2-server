from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    rgb_topic_arg     = DeclareLaunchArgument('rgb_topic',     description='RGB image topic name')
    s1_model_path_arg = DeclareLaunchArgument('s1_model_path', description='System1 model path')
    s2_model_path_arg = DeclareLaunchArgument('s2_model_path', description='System2 model path')
    s1_device_arg     = DeclareLaunchArgument('s1_device', default_value='cuda:0', description='System1 CUDA device')
    s2_device_arg     = DeclareLaunchArgument('s2_device', default_value='cuda:1', description='System2 CUDA device')

    rgb_topic     = LaunchConfiguration('rgb_topic')
    s1_model_path = LaunchConfiguration('s1_model_path')
    s2_model_path = LaunchConfiguration('s2_model_path')
    s1_device     = LaunchConfiguration('s1_device')
    s2_device     = LaunchConfiguration('s2_device')

    system1_node = Node(
        package='internnav_server',
        executable='system1',
        name='internnav_system1',
        output='screen',
        parameters=[{
            'rgb_topic': rgb_topic,
            'model_path': s1_model_path,
            'device': s1_device,
        }],
    )

    system2_node = Node(
        package='internnav_server',
        executable='system2',
        name='internnav_system2',
        output='screen',
        parameters=[{
            'rgb_topic': rgb_topic,
            'model_path': s2_model_path,
            'device': s2_device,
        }],
    )

    return LaunchDescription([
        rgb_topic_arg,
        s1_model_path_arg,
        s2_model_path_arg,
        s1_device_arg,
        s2_device_arg,
        system1_node,
        system2_node,
    ])
