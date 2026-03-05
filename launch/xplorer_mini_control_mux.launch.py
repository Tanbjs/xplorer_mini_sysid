import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    signal_config = os.path.join(
        get_package_share_directory('xplorer_mini_sysid'),
            'config',
            'signal_config.yaml' )

    auv_param = os.path.join(
        get_package_share_directory('xplorer_mini_descriptions'),
            'robots',
            'xplorer_mini_dynamic_parameters.yaml')
    
    ocean_current_param = os.path.join(
        get_package_share_directory('xplorer_mini_sysid'),
            'config',
            'ocean_current_config.yaml')    

    node=Node(
        package = 'xplorer_mini_sysid',
        name = 'control_mux',
        namespace='xplorer_mini',
        executable = 'control_mux',
        output="screen",
        emulate_tty=True,
        parameters = [signal_config, auv_param, ocean_current_param]
    )
    
    ld.add_action(node)
    return ld