import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (LogInfo, RegisterEventHandler)
from launch.event_handlers import (OnProcessStart)
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def generate_launch_description():
    
    model = 'edmdc'                 # 'dmdc' or 'edmdc'
    pose_controller = 'ffpi'        # 'ffpi' or 'pi'
    vel_controller = 'intmpc'       # 'pid' or 'kmpc' (model-based MPC)
    constrained = True              # Whether to use constrained MPC (only for kmpc)
    use_preview = True              # Whether to use preview control (only for kmpc)


    auv_param = os.path.join(
        get_package_share_directory('xplorer_mini_descriptions'),
        'robots',
        'xplorer_mini_dynamic_parameters.yaml'
    )

    dpid_gain = os.path.join(
        get_package_share_directory('xplorer_mini_sysid'),
        'config',
        'controller',
        'dpid_gain.yaml'
    )  
    
    gain = os.path.join(
        get_package_share_directory('xplorer_mini_sysid'),
        'config', 
        'controller', 
        model,
        'constrained' if constrained else 'unconstrained',
        'with_preview' if use_preview else 'without_preview',
        f'{pose_controller}_{vel_controller}_gain.yaml'
    )

    control_node = Node(
        package='xplorer_mini_sysid',
        executable='kmpc.py',
        namespace='xplorer_mini',
        name='kmpc_node',
        output='screen',
        emulate_tty=True,
        parameters=[gain, auv_param , {'use_sim_time': True}]
    )

    return LaunchDescription([
                            control_node,
                            ])