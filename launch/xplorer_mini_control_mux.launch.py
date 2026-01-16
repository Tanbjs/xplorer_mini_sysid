import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    signal_config = os.path.join(
        get_package_share_directory('xplorer_mini_sysid'),
            'config',
            'signal_config.yaml'
        )
        
    node=Node(
        package = 'xplorer_mini_control',
        name = 'xplorer_mini_control_node',
        namespace='xplorer_mini',
        executable = 'xplorer_mini_robust_control',
        output="screen",
        emulate_tty=True,
        parameters = [signal_config]
    )

    ld.add_action(node)
    return ld