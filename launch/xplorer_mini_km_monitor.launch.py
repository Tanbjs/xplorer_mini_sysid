import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()
    
    koopman_monitor_config = os.path.join(
        get_package_share_directory('xplorer_mini_sysid'),
        'config',
        'koopman_monitor_config.yaml'
    )

    monitor_node = Node(
        package='xplorer_mini_sysid',
        name='km_monitor',
        namespace='xplorer_mini',
        executable='km_monitor.py',
        output="screen",
        emulate_tty=True,
        parameters=[koopman_monitor_config, {'use_sim_time': True}]
    )

    ld.add_action(monitor_node)
    return ld