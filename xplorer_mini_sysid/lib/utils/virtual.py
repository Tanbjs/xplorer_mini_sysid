import numpy as np

from xplorer_mini_sysid.lib.utils.kinematic import cal_eta_err_with_ssa, eulerang
from .controller import PositionControllerType, VelocityControllerType


def generate_virtual_reference_pid(
    eta_ref_window: np.ndarray, 
    eta: np.ndarray, 
    nu_cmd_b: np.ndarray, 
    N_horizon: int, 
    dt: float, 
    pose_ct_virtual: PositionControllerType, 
    pose_ct_actual: PositionControllerType, 
):
    # Preallocate 
    nu_hat_cmd_b_window = np.zeros((N_horizon, 6))
    eta_hat = np.copy(eta)
    nu_hat_cmd_b_window[0] = nu_cmd_b

    # Inject outer loop integral and derivative states for continuity
    pose_ct_virtual.integral = np.copy(pose_ct_actual.integral)
    pose_ct_virtual.prev_error = np.copy(pose_ct_actual.prev_error)
    
    for i in range(1, N_horizon):
        nu_hat_cmd_b = pose_ct_virtual.compute_control(eta_hat, eta_ref_window[i], dt)
        J, _, _ = eulerang(eta_hat[3], eta_hat[4], eta_hat[5])
        
        nu_hat_cmd_b_window[i] = nu_hat_cmd_b
        eta_hat = eta_hat + dt * (J @ nu_hat_cmd_b)

    return nu_hat_cmd_b_window

def generate_virtual_reference_ff_pi(
    eta_ref_window:np.ndarray, 
    eta: np.ndarray, 
    nu_cmd_b: np.ndarray, 
    use_filter: bool, 
    eta_dot_ref_filtered: np.ndarray, 
    alpha_ff: float, 
    N_horizon: int, 
    dt: float, 
    pose_ct_actual: PositionControllerType,
    pose_ct_virtual: PositionControllerType, 
    use_feedforward: bool
):
    # Numerical differentiation for Feed-forward (dot{eta}_ref)
    if use_filter:
        last_filt_val = np.copy(eta_dot_ref_filtered) 
        alpha = alpha_ff
        eta_dot_ref_window = np.zeros_like(eta_ref_window)
        for i in range(1, N_horizon):
            eta_dot_ref_raw_diff = cal_eta_err_with_ssa(eta_ref_window[i], eta_ref_window[i-1]) / dt
            eta_dot_ref_window[i] = (alpha * eta_dot_ref_raw_diff) + (1.0 - alpha) * last_filt_val
            last_filt_val = eta_dot_ref_window[i]
    else: 
        eta_dot_ref_window = np.gradient(eta_ref_window, dt, axis=0, edge_order=1)

    nu_hat_cmd_b_window = np.zeros((N_horizon, 6))
    
    # Clone current state for virtual reference simulation
    eta_hat = np.copy(eta)
    pose_ct_virtual.integral = np.copy(pose_ct_actual.integral)
    
    # Initial step
    nu_hat_cmd_b_window[0] = nu_cmd_b

    for i in range(1, N_horizon):
        eta_ref = eta_ref_window[i]
        v_ff = eta_dot_ref_window[i] if use_feedforward else np.zeros(6)
        
        # Predict next velocity in Body Frame
        nu_cmd_b = pose_ct_virtual.compute_control(eta_hat, eta_ref, dt, v_ff)
        nu_hat_cmd_b_window[i] = nu_cmd_b
        
        # Kinematics Update: Convert Body Velocity to NED Velocity
        J, _, _ = eulerang(eta_hat[3], eta_hat[4], eta_hat[5])
        # J, _, _ = eulerang(eta_ref[3], eta_ref[4], eta_ref[5])  # Use reference orientation for feedforward consistency
        eta_dot_hat = J @ nu_cmd_b
        
        # Update Virtual Pose (Forward Euler)
        eta_hat = eta_hat + (dt * eta_dot_hat)

    return nu_hat_cmd_b_window