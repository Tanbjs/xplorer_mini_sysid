import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (LogInfo, RegisterEventHandler)
from launch.event_handlers import (OnProcessStart)
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():

    thruster_data = os.path.join(
        get_package_share_directory('xplorer_mini_descriptions'),
        'robots',
        'bluerobotics_t200_data.yaml'
    )

    thruster_config = os.path.join(
        get_package_share_directory('xplorer_mini_descriptions'),
        'robots',
        'xplorer_mini_thruster_configuration.yaml'
    )

    kmpc_config = os.path.join(
        get_package_share_directory('xplorer_mini_sysid'),
        'config',
        'kmpc_config.yaml'
    )
    
    signal_config = os.path.join(
        get_package_share_directory('xplorer_mini_sysid'),
            'config',
            'signal_config.yaml'
        )
    
    thruster_manager = Node(
        package='xplorer_mini_thruster_manager',
        executable='thruster_manager',
        namespace='xplorer_mini',
        name='gnc_thruster_manager',
        output='screen',
        emulate_tty=True,
        parameters=[{thruster_config}, {thruster_data}, {'use_sim_time': True}]
    )

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
        emulate_tty=True)

    reference_model = Node(
        package='xplorer_mini_guidance',
        executable='reference_model',
        namespace='xplorer_mini',
        name='gnc_reference_model',
        output='screen',
        parameters=[{'use_sim_time': True}],
        emulate_tty=True)

    teleop_twist_to_pose = Node(
        package='xplorer_mini_guidance',
        executable='ref_twist_to_pose',
        namespace='xplorer_mini',   
        name='gnc_ref_twist_to_pose',
        output='screen')

    move_to_pose = Node(
        package='xplorer_mini_guidance',
        executable='move_to_pose_action_server',
        namespace='xplorer_mini',
        name='move_to_pose_action_server_node',
        output='screen')
    
    control_node = Node(
        package = 'xplorer_mini_sysid',
        name = 'kmpc_node',
        namespace='xplorer_mini',
        executable = 'kmpc.py',
        output="screen",
        emulate_tty=True,
        parameters = [{kmpc_config}, {'use_sim_time': True}]
    )

    auv_state_sync = Node(
        package='xplorer_mini_cpp_utils',
        executable='auv_state_sync',
        namespace='xplorer_mini',
        name='gnc_auv_state_sync',
        output='screen',
        parameters=[{'use_sim_time': True}],
        emulate_tty=True)
    
    control_mux = Node(
        package = 'xplorer_mini_sysid',
        name = 'control_mux',
        namespace='xplorer_mini',
        executable = 'control_mux',
        output="screen",
        emulate_tty=True,
        parameters = [{signal_config}, {'use_sim_time': True}]
    )

    # xplorer_mini_thruster_manager_dir = get_package_share_directory('xplorer_mini_thruster_manager')
    # thruster_manager_launch = IncludeLaunchDescription(PythonLaunchDescriptionSource(
    #                 xplorer_mini_thruster_manager_dir + '/launch/thruster_manager.launch.py'))

    return LaunchDescription([reference_model,
                            reference_traj_mux,
                            auv_state_sync,
                            teleop_twist_to_pose,
                            move_to_pose,
                            reference_traj_gen,
                            control_mux,
                            RegisterEventHandler(
                                OnProcessStart(
                                target_action=reference_model,
                                on_start=[
                                    LogInfo(msg='Reference Trajectory Generator is started, Starting the controller'),
                                    thruster_manager,
                                    control_node
                                ]
                            ))
                        ])