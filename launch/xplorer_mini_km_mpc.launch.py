import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (LogInfo, RegisterEventHandler)
from launch.event_handlers import (OnProcessStart)
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():

    auv_param = os.path.join(
        get_package_share_directory('xplorer_mini_descriptions'),
            'robots',
            'xplorer_mini_dynamic_parameters.yaml'
        )

    kmpc_config = os.path.join(
        get_package_share_directory('xplorer_mini_sysid'),
        'config',
        'kmpc_config.yaml'
    )
    
    control_node = Node(
        package='xplorer_mini_sysid',
        executable='kmpc.py',
        namespace='xplorer_mini',
        name='kmpc_node',
        output='screen',
        emulate_tty=True,
        parameters=[{kmpc_config}, {auv_param}, {'use_sim_time': True}]
    )

    return LaunchDescription([
                            control_node,
                            ])