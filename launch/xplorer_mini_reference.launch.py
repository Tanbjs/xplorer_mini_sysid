import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration as Lc
from launch.actions import DeclareLaunchArgument


def launch_setup(context, *args, **kwargs):

    reference_traj_mux = Node(
        package='xplorer_mini_guidance',
        executable='ref_trajectory_mux',
        namespace='xplorer_mini',
        name='gnc_ref_trajectory_mux',
        output='screen',
        emulate_tty=True)
    
    reference_traj_gen = Node(
        package='xplorer_mini_guidance',
        executable='ref_trajectory_generation',
        namespace='xplorer_mini',
        name='gnc_ref_trajectory_generation',
        output='screen',
        parameters=[{'use_sim_time': True}],
        emulate_tty=True)

    reference_model = Node(
        package='xplorer_mini_guidance',
        executable='reference_model',
        namespace='xplorer_mini',
        name='gnc_reference_model',
        output='screen',
        parameters=[{'use_sim_time': True}],
        emulate_tty=True)

    return [
            reference_traj_gen, 
            reference_traj_mux,
            reference_model
        ]


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])
