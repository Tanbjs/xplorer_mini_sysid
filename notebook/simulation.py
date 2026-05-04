import os
import mlflow
import yaml
from pathlib import Path
import numpy as np
import warnings
import urllib3
from matplotlib import cm
import matplotlib.pyplot as plt

from xplorer_mini_guidance.utils.path_gen import figure8_path, circle_path, zigzag_path
from xplorer_mini_python_utils.kinematics import eulerang, gvect, m2c

from xplorer_mini_sysid.lib.utils.mlflow import load_model
from xplorer_mini_sysid.lib.utils.kinematic import cal_eta_err_with_ssa
from xplorer_mini_sysid.lib.utils.controller import create_position_controller, create_velocity_controller
from xplorer_mini_sysid.lib.utils.virtual import generate_virtual_reference_ff_pi, generate_virtual_reference_pid

import logging
warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
sim_logger = logging.getLogger("Simulation")
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

class UUVParams:
    def __init__(self, yaml_data):
        raw = yaml_data['/**']['ros__parameters']
        self.W, self.B, self.m = raw['weight'], raw['buoyancy'], raw['mass']
        self.r_g, self.r_b = np.array(raw['r_g']), np.array(raw['r_b'])
        self.M_total = np.array(raw['rigid_body_mass']).reshape(6, 6) + np.array(raw['added_mass']).reshape(6, 6)
        self.D_l = np.array(raw['linear_drag']).reshape(6, 6)
        self.D_nl = np.array(raw['nonlinear_drag']).reshape(6, 6)

class MockParam:
    """Mock Object เลียนแบบ rclpy.parameter.Parameter"""
    def __init__(self, value):
        self.value = value

def create_mock_ros_params(param_dict, parent_key=''):
    """แปลงและแผ่ (Flatten) Dictionary ให้เหมือน ROS 2 Parameters แบบ Dot Notation"""
    mocked_dict = {}
    for k, v in param_dict.items():
        full_key = f"{parent_key}{k}"
        if isinstance(v, dict):
            # แผ่ Nested Dict เป็น flat keys (เช่น 'params' -> 'params.kp')
            mocked_dict.update(create_mock_ros_params(v, f"{full_key}."))
        else:
            mocked_dict[full_key] = MockParam(v)
    return mocked_dict

def load_params(param_file):
    with open(param_file, 'r') as f: return UUVParams(yaml.safe_load(f))

def get_effective_buoyancy(z, B_max, robot_height=0.5):
    """
    NED Coordinate: Z points downward (Z=0 is water surface, Z>0 is underwater)
    """
    top_z = z - (robot_height / 2.0)     # Top of the robot (lower Z value)
    bottom_z = z + (robot_height / 2.0)  # Bottom of the robot (higher Z value)

    if top_z >= 0.0:
        return B_max  # Fully submerged
    elif bottom_z <= 0.0:
        return 0.0    # Fully emerged
    else:
        submerged_ratio = bottom_z / robot_height
        return B_max * submerged_ratio
    
def f_dyn(x, u, params: UUVParams):
    eta, nu = x[0:6], x[6:12]
    B_eff = get_effective_buoyancy(eta[2], params.B, robot_height=0.3)
    J_eta, _, _ = eulerang(eta[3], eta[4], eta[5])
    sum_forces = u - (m2c(params.M_total, nu) @ nu) - ((params.D_l + params.D_nl @ np.diag(np.abs(nu))) @ nu) - gvect(params.W, B_eff, eta[4], eta[3], params.r_g, params.r_b)
    return np.concatenate(((J_eta @ nu).flatten(), np.linalg.solve(params.M_total, sum_forces).flatten()))

def rk4_step(sys, x, u, dt):
    k1 = sys(x, u)
    k2 = sys(x + 0.5 * dt * k1, u)
    k3 = sys(x + 0.5 * dt * k2, u)
    k4 = sys(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def plot_response(histories, labels, eta_ref_full, save_dir=None):

    # ================= Data Sorting & Q1 Paper Colors =================
    def get_config(lbl):
        lbl_lower = str(lbl).lower()

        # 1. Controller Priority: pid (0), nominal (1), offset-free (2)
        if 'pid' in lbl_lower: ctrl_p = 0
        elif 'nominal' in lbl_lower: ctrl_p = 1
        elif 'offset-free' in lbl_lower: ctrl_p = 2
        else: ctrl_p = 3

        # 2. Model Priority: dmdc (0), edmdc (1)
        if 'edmdc' in lbl_lower: mod_p = 1
        elif 'dmdc' in lbl_lower: mod_p = 0
        else: mod_p = 2

        # 3. Preview Priority: without (0), with (1)
        if 'with preview' in lbl_lower: prev_p = 1
        else: prev_p = 0

        keys = (ctrl_p, mod_p, prev_p)

        # ---- Color (Cool bright · paired by preview) ----
        # without preview = light shade, with preview = deeper shade (still soft cool)
        if ctrl_p == 0:
            color = '#758A93'      # Soft gray (PID baseline)
        elif ctrl_p == 1 and mod_p == 0 and prev_p == 0:
            color = '#7FCBC4'      # Light teal     (Nominal, DMDc, w/o)
        elif ctrl_p == 1 and mod_p == 0 and prev_p == 1:
            color = '#2A9D8F'      # Deeper teal    (Nominal, DMDc, w/)
        elif ctrl_p == 1 and mod_p == 1 and prev_p == 0:
            color = '#8FB8DE'      # Light sky blue (Nominal, eDMDc, w/o)
        elif ctrl_p == 1 and mod_p == 1 and prev_p == 1:
            color = '#3A78B8'      # Deeper blue    (Nominal, eDMDc, w/)
        elif ctrl_p == 2 and mod_p == 0 and prev_p == 0:
            color = '#CC6B5C'      # Light red       (Offset-Free, DMDc, w/o)
        elif ctrl_p == 2 and mod_p == 0 and prev_p == 1:
            color = '#C5172E'      # Soft terracotta (Offset-Free, DMDc, w/) C5172E C0392B
        elif ctrl_p == 2 and mod_p == 1 and prev_p == 0:
            color = '#D9A055'      # Light orange    (Offset-Free, eDMDc, w/o)
        elif ctrl_p == 2 and mod_p == 1 and prev_p == 1:
            color = '#FF7444'      # Soft amber      (Offset-Free, eDMDc, w/)
        else:
            color = '#B0B0B0'      # Gray fallback

        # ---- Line style (all solid; preview distinguished by line width) ----
        linestyle = '-'

        # ---- Line width (PID slightly thicker as baseline; others uniform) ----
        linewidth = 2.0 if ctrl_p == 0 else 1.5

        return keys, color, linestyle, linewidth

    # Sort by (Controller -> Model -> Preview)
    sorted_data = sorted(zip(labels, histories), key=lambda x: get_config(x[0])[0])
    labels = [x[0] for x in sorted_data]
    histories = [x[1] for x in sorted_data]

    # Locked style per label
    styles = [get_config(lbl) for lbl in labels]
    colors     = [s[1] for s in styles]
    linestyles = [s[2] for s in styles]
    linewidths = [s[3] for s in styles]
    # =================================================================

    t = np.array(histories[0]['t'])
    eta_ref = eta_ref_full[:len(t), :]

    # Optimization: Convert all data to numpy arrays once to reduce CPU overhead
    for h in histories:
        if not isinstance(h['x'], np.ndarray):
            h['x'] = np.array(h['x'])
        if 'nu_cmd_b' in h and not isinstance(h['nu_cmd_b'], np.ndarray):
            h['nu_cmd_b'] = np.array(h['nu_cmd_b'])
        if 'tau' in h and not isinstance(h['tau'], np.ndarray):
            h['tau'] = np.array(h['tau'])

    def format_figure(fig, axs, bottom_margin):
        # Align y-labels vertically
        fig.align_ylabels(axs)

        handles, labels_l = axs[0, 0].get_legend_handles_labels()
        fig.tight_layout(rect=[0.02, bottom_margin, 1, 0.96])
        fig.legend(handles, labels_l, loc='upper center', ncol=3,
                   bbox_to_anchor=(0.5, bottom_margin), columnspacing=2.0,
                   handletextpad=0.5, fontsize=9, frameon=True)

    # ================= Figure 1: Position Response =================
    fig1, axs1 = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    fig1.suptitle(r'Position Tracking ($\eta$)', fontsize=14, fontweight='bold')
    labels_eta_lin = ['x [m]', 'y [m]', 'z [m]']
    labels_eta_ang = [r'$\phi$ [deg]', r'$\theta$ [deg]', r'$\psi$ [deg]']

    for i in range(3):
        ref_ang_rad = np.unwrap(eta_ref[:, i+3]) if i == 2 else eta_ref[:, i+3]
        axs1[i, 0].plot(t, eta_ref[:, i], 'k--', linewidth=2, label='Ref' if i == 0 else "")
        axs1[i, 1].plot(t, np.rad2deg(ref_ang_rad), 'k--', linewidth=2, label='Ref' if i == 0 else "")

        for idx, history in enumerate(histories):
            eta = history['x'][:, 0:6]
            state_ang_rad = np.unwrap(eta[:, i+3]) if i == 2 else eta[:, i+3]

            axs1[i, 0].plot(t, eta[:, i],
                            color=colors[idx], linestyle=linestyles[idx], linewidth=linewidths[idx],
                            label=labels[idx] if i == 0 else "")
            axs1[i, 1].plot(t, np.rad2deg(state_ang_rad),
                            color=colors[idx], linestyle=linestyles[idx], linewidth=linewidths[idx],
                            label=labels[idx] if i == 0 else "")

        axs1[i, 0].set_ylabel(labels_eta_lin[i])
        axs1[i, 0].grid(True, linestyle=':', alpha=0.7)
        axs1[i, 1].set_ylabel(labels_eta_ang[i])
        axs1[i, 1].grid(True, linestyle=':', alpha=0.7)

    axs1[2, 0].set_xlabel('Time [s]')
    axs1[2, 1].set_xlabel('Time [s]')
    format_figure(fig1, axs1, bottom_margin=0.11)

    # ================= Figure 2: Position Tracking Error =================
    fig2, axs2 = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    fig2.suptitle(r'Position Tracking Error ($e_\eta$)', fontsize=14, fontweight='bold')
    labels_eeta_lin = [r'$e_x$ [m]', r'$e_y$ [m]', r'$e_z$ [m]']
    labels_eeta_ang = [r'$e_\phi$ [deg]', r'$e_\theta$ [deg]', r'$e_\psi$ [deg]']

    for i in range(3):
        for idx, history in enumerate(histories):
            eta = history['x'][:, 0:6]

            e_lin = eta_ref[:, i] - eta[:, i]
            # Shortest angular difference preventing wrap-around artifacts
            e_ang_rad = (eta_ref[:, i+3] - eta[:, i+3] + np.pi) % (2 * np.pi) - np.pi

            axs2[i, 0].plot(t, e_lin,
                            color=colors[idx], linestyle=linestyles[idx], linewidth=linewidths[idx],
                            label=labels[idx] if i == 0 else "")
            axs2[i, 1].plot(t, np.rad2deg(e_ang_rad),
                            color=colors[idx], linestyle=linestyles[idx], linewidth=linewidths[idx],
                            label=labels[idx] if i == 0 else "")

        axs2[i, 0].set_ylabel(labels_eeta_lin[i])
        axs2[i, 0].grid(True, linestyle=':', alpha=0.7)
        axs2[i, 1].set_ylabel(labels_eeta_ang[i])
        axs2[i, 1].grid(True, linestyle=':', alpha=0.7)

    axs2[2, 0].set_xlabel('Time [s]')
    axs2[2, 1].set_xlabel('Time [s]')
    format_figure(fig2, axs2, bottom_margin=0.13)

    # ================= Figure 3: Velocity Response =================
    fig3, axs3 = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    fig3.suptitle(r'Velocity Tracking ($\nu$)', fontsize=14, fontweight='bold')
    labels_nu_lin = ['u [m/s]', 'v [m/s]', 'w [m/s]']
    labels_nu_ang = ['p [deg/s]', 'q [deg/s]', 'r [deg/s]']

    for i in range(3):
        for idx, history in enumerate(histories):
            nu = history['x'][:, 6:12]
            nu_cmd = history['nu_cmd_b']

            nu_deg = nu.copy()
            nu_deg[:, 3:6] = np.rad2deg(nu_deg[:, 3:6])
            nu_cmd_deg = nu_cmd.copy()
            nu_cmd_deg[:, 3:6] = np.rad2deg(nu_cmd_deg[:, 3:6])

            # Command: lighter / dotted, Actual: solid|dashed per preview
            axs3[i, 0].plot(t, nu_cmd[:, i],
                            color=colors[idx], linestyle='--', linewidth=1.0, alpha=0.5,
                            label=f'{labels[idx]} (Cmd)' if i == 0 else "")
            axs3[i, 0].plot(t, nu[:, i],
                            color=colors[idx], linestyle=linestyles[idx], linewidth=linewidths[idx],
                            label=labels[idx] if i == 0 else "")
            axs3[i, 1].plot(t, nu_cmd_deg[:, i+3],
                            color=colors[idx], linestyle='--', linewidth=1.0, alpha=0.5)
            axs3[i, 1].plot(t, nu_deg[:, i+3],
                            color=colors[idx], linestyle=linestyles[idx], linewidth=linewidths[idx])

        axs3[i, 0].set_ylabel(labels_nu_lin[i])
        axs3[i, 0].grid(True, linestyle=':', alpha=0.7)
        axs3[i, 1].set_ylabel(labels_nu_ang[i])
        axs3[i, 1].grid(True, linestyle=':', alpha=0.7)

    axs3[2, 0].set_xlabel('Time [s]')
    axs3[2, 1].set_xlabel('Time [s]')
    format_figure(fig3, axs3, bottom_margin=0.18)

    # ================= Figure 4: Velocity Error =================
    fig4, axs4 = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    fig4.suptitle(r'Velocity Tracking Error ($e_\nu$)', fontsize=14, fontweight='bold')
    labels_enu_lin = [r'$e_u$ [m/s]', r'$e_v$ [m/s]', r'$e_w$ [m/s]']
    labels_enu_ang = [r'$e_p$ [deg/s]', r'$e_q$ [deg/s]', r'$e_r$ [deg/s]']

    for i in range(3):
        for idx, history in enumerate(histories):
            nu = history['x'][:, 6:12]
            nu_cmd = history['nu_cmd_b']
            e_nu_lin = nu_cmd[:, i] - nu[:, i]
            e_nu_ang = np.rad2deg(nu_cmd[:, i+3] - nu[:, i+3])

            axs4[i, 0].plot(t, e_nu_lin,
                            color=colors[idx], linestyle=linestyles[idx], linewidth=linewidths[idx],
                            label=labels[idx] if i == 0 else "")
            axs4[i, 1].plot(t, e_nu_ang,
                            color=colors[idx], linestyle=linestyles[idx], linewidth=linewidths[idx],
                            label=labels[idx] if i == 0 else "")

        axs4[i, 0].set_ylabel(labels_enu_lin[i])
        axs4[i, 0].grid(True, linestyle=':', alpha=0.7)
        axs4[i, 1].set_ylabel(labels_enu_ang[i])
        axs4[i, 1].grid(True, linestyle=':', alpha=0.7)

    axs4[2, 0].set_xlabel('Time [s]')
    axs4[2, 1].set_xlabel('Time [s]')
    format_figure(fig4, axs4, bottom_margin=0.13)

    # ================= Figure 5: Generalized Torque =================
    fig5, axs5 = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    fig5.suptitle(r'Generalized Torque ($\tau$)', fontsize=14, fontweight='bold')
    labels_tau_f, labels_tau_t = ['X [N]', 'Y [N]', 'Z [N]'], ['K [Nm]', 'M [Nm]', 'N [Nm]']

    for i in range(3):
        for idx, history in enumerate(histories):
            tau = history['tau']
            axs5[i, 0].plot(t, tau[:, i],
                            color=colors[idx], linestyle=linestyles[idx], linewidth=linewidths[idx],
                            label=labels[idx] if i == 0 else "")
            axs5[i, 1].plot(t, tau[:, i+3],
                            color=colors[idx], linestyle=linestyles[idx], linewidth=linewidths[idx])

        axs5[i, 0].set_ylabel(labels_tau_f[i])
        axs5[i, 0].grid(True, linestyle=':', alpha=0.7)
        axs5[i, 1].set_ylabel(labels_tau_t[i])
        axs5[i, 1].grid(True, linestyle=':', alpha=0.7)

    axs5[2, 0].set_xlabel('Time [s]')
    axs5[2, 1].set_xlabel('Time [s]')
    format_figure(fig5, axs5, bottom_margin=0.11)

    # ================= Figure 6: 3D Path =================
    fig6 = plt.figure(figsize=(10, 10))
    ax6 = fig6.add_subplot(111, projection='3d')

    ax6.plot(eta_ref[:, 0], eta_ref[:, 1], eta_ref[:, 2], 'k--', linewidth=2, label='Reference')

    for idx, history in enumerate(histories):
        eta = history['x'][:, 0:6]
        ax6.plot(eta[:, 0], eta[:, 1], eta[:, 2],
                 color=colors[idx], linestyle=linestyles[idx], linewidth=linewidths[idx],
                 label=labels[idx])

    ax6.set_xlabel('x [m]')
    ax6.set_ylabel('y [m]')
    ax6.set_zlabel('z [m]')
    ax6.set_title('3D Path Tracking Comparison', fontsize=14, fontweight='bold', pad=20)

    handles6, labels6 = ax6.get_legend_handles_labels()
    fig6.tight_layout(rect=[0, 0.11, 1, 1])
    fig6.legend(handles6, labels6, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.11),
                columnspacing=2.0, fontsize=10, frameon=True)

    # ================= Metrics & Tables =================
    metrics_eta, metrics_nu = [], []
    for history in histories:
        eta = history['x'][:, 0:6]
        e_lin = eta_ref[:, :3] - eta[:, :3]
        e_ang_rad = (eta_ref[:, 3:6] - eta[:, 3:6] + np.pi) % (2 * np.pi) - np.pi
        e_eta = np.hstack((e_lin, np.rad2deg(e_ang_rad)))
        metrics_eta.append({'rmse': np.sqrt(np.mean(e_eta**2, axis=0)),
                            'mae': np.mean(np.abs(e_eta), axis=0),
                            'maxae': np.max(np.abs(e_eta), axis=0)})

        nu = history['x'][:, 6:12]
        nu_cmd = history['nu_cmd_b']
        e_nu = np.hstack((nu_cmd[:, :3] - nu[:, :3], np.rad2deg(nu_cmd[:, 3:6] - nu[:, 3:6])))
        metrics_nu.append({'rmse': np.sqrt(np.mean(e_nu**2, axis=0)),
                           'mae': np.mean(np.abs(e_nu), axis=0),
                           'maxae': np.max(np.abs(e_nu), axis=0)})

    def create_metric_figure(title_main, cols, metrics_data):
        fig, axs = plt.subplots(3, 1, figsize=(14, 8))
        fig.suptitle(title_main, fontweight='bold', fontsize=14)
        metric_keys, metric_titles = ['rmse', 'mae', 'maxae'], ['RMSE', 'MAE', 'MaxAE']
        colors_bg = ['#d9ead3', '#cfe2f3', '#f4cccc']
        highlight_color, highlight_text_color = '#fff2cc', '#d62728'
        for i, key in enumerate(metric_keys):
            cell_text = [[f"{m[key][col_idx]:.4f}" for col_idx in range(6)] for m in metrics_data]
            tbl = axs[i].table(cellText=cell_text, rowLabels=labels, colLabels=cols,
                               loc='center', cellLoc='center',
                               rowColours=['#f2f2f2']*len(labels), colColours=[colors_bg[i]]*6)
            for col_idx in range(6):
                col_vals = [m[key][col_idx] for m in metrics_data]
                min_idx = np.argmin(col_vals)
                best_cell = tbl[min_idx + 1, col_idx]
                best_cell.set_facecolor(highlight_color)
                best_cell.get_text().set_weight('bold')
                best_cell.get_text().set_color(highlight_text_color)
            tbl.scale(1, 1.8)
            tbl.set_fontsize(11)
            axs[i].axis('off')
            axs[i].set_title(metric_titles[i], pad=5, fontweight='bold')
        fig.tight_layout()
        return fig

    fig7 = create_metric_figure(r'$\eta$ Error Metrics',
                                ['x [m]', 'y [m]', 'z [m]', r'$\phi$ [deg]', r'$\theta$ [deg]', r'$\psi$ [deg]'],
                                metrics_eta)
    fig8 = create_metric_figure(r'$\nu$ Error Metrics',
                                ['u [m/s]', 'v [m/s]', 'w [m/s]', 'p [deg/s]', 'q [deg/s]', 'r [deg/s]'],
                                metrics_nu)

    # ================= Save or Show =================
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8]
        names = ["1_position_response",
                 "2_position_error",
                 "3_velocity_response",
                 "4_velocity_error",
                 "5_control_effort",
                 "6_3d_path",
                 "7_table_position_metrics",
                 "8_table_velocity_metrics"]
        for f, name in zip(figs, names):
            f.savefig(save_path / f"{name}.svg", format='svg', bbox_inches='tight')
        plt.close('all')
    else:
        plt.show()

def get_config_flags(ros_params):
    pose_params = ros_params['position_controller']
    vel_params = ros_params['velocity_controller']
    return {
        'pose_ctrl_type': pose_params.get('type', 'pid'),
        'use_ff':         pose_params.get('use_feedforward', False),
        'use_filter':     pose_params.get('use_filter', False),
        'alpha_ff':       pose_params.get('alpha_ff', 0.8),
        'vel_ctrl_type':  vel_params.get('type', 'pid'),
        'use_preview':    vel_params.get('use_preview', False),
        'N_horizon':      vel_params.get('params', {}).get('N_horizon', 10)
    }

def init_controllers(koopman_model, ros_params, node_name, sim_logger):
    pose_params_native = ros_params['position_controller']
    vel_params_native = ros_params['velocity_controller']
    
    pose_ctrl = create_position_controller(logger=sim_logger, **create_mock_ros_params(pose_params_native))
    vel_ctrl  = create_velocity_controller(model=koopman_model, node_name=node_name, dt=0.1, logger=sim_logger, **create_mock_ros_params(vel_params_native))
    
    return pose_ctrl, vel_ctrl

def run_velocity_simulation(auv_params, vel_ctrl, nu_ref_full, config_flags, t_end=120.0, dt=0.1, DEBUG_MODE=False):
    num_steps = int(t_end / dt)
    history = {'t': [], 'x': [], 'tau': [], 'nu_cmd_b': []}
    T_ned_nwu = np.diag([1.0, -1.0, -1.0, 1.0, -1.0, -1.0])

    # 1. Extract Controller Flags
    N_horizon     = config_flags.get('N_horizon', 10)
    vel_ctrl_type = config_flags.get('vel_ctrl_type', 'mpc')

    sys = lambda x, u: f_dyn(x, u, auv_params)
    
    # 2. Initial State Setup: [eta (6), nu (6)]
    if DEBUG_MODE:
        x_curr = np.concatenate((np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0]), np.zeros(6)))
    else:
        # Start at origin, initial velocity matches the first reference
        x_curr = np.concatenate((np.zeros(6), nu_ref_full[0,:].flatten()))

    # Initial Log
    history['t'].append(0.0)
    history['x'].append(x_curr.copy())
    history['tau'].append(np.zeros(6))
    history['nu_cmd_b'].append(nu_ref_full[0,:].copy())

    # 3. Simulation Loop
    for i in range(num_steps):
        eta, nu = x_curr[0:6], x_curr[6:12]

        # Get Velocity Reference Window (B-Frame)
        end_idx = min(i + N_horizon, len(nu_ref_full))
        nu_ref_window = nu_ref_full[i : end_idx, :]
        
        if len(nu_ref_window) < N_horizon:
            padding = np.repeat([nu_ref_window[-1]], N_horizon - len(nu_ref_window), axis=0)
            nu_ref_window = np.vstack((nu_ref_window, padding))

        nu_cmd_b = nu_ref_window[0]

        # --- A. Transformations (NED -> NWU) ---
        nu_nwu = T_ned_nwu @ nu
        nu_cmd_b_nwu = T_ned_nwu @ nu_cmd_b

        # --- B. Velocity Control (NWU) ---
        if 'mpc' in vel_ctrl_type:
            # Pass the entire NWU reference window if MPC supports preview
            nu_ref_window_nwu = nu_ref_window @ T_ned_nwu.T
            tau_cmd_nwu = vel_ctrl.compute_control(nu_nwu, nu_ref_window_nwu)
        else:
            tau_cmd_nwu = vel_ctrl.compute_control(nu_nwu, nu_cmd_b_nwu, dt)

        # --- C. Transformation (NWU -> NED) & Actuation Limits ---
        tau_cmd = T_ned_nwu @ np.asarray(tau_cmd_nwu).flatten()
        tau_cmd = np.clip(tau_cmd, -200, 200)
        
        # --- D. Plant Dynamics Integration ---
        x_curr = rk4_step(sys, x_curr, tau_cmd, dt)
        x_curr[3:6] = (x_curr[3:6] + np.pi) % (2 * np.pi) - np.pi

        # --- E. Logging ---
        history['t'].append((i + 1) * dt)
        history['x'].append(x_curr.copy())
        history['tau'].append(tau_cmd.copy())
        history['nu_cmd_b'].append(nu_cmd_b.copy())

    return history

def run_cascade_simulation(auv_params, pose_ctrl, vel_ctrl, eta_ref_full, config_flags, t_end=120.0, dt=0.1, DEBUG_MODE=False):
    num_steps = int(t_end / dt)
    history = {'t': [], 'x': [], 'tau': [], 'nu_cmd_b': []}
    T_ned_nwu = np.diag([1.0, -1.0, -1.0, 1.0, -1.0, -1.0])

    # 1. Extract Controller Flags
    N_horizon      = config_flags.get('N_horizon', 10)
    use_ff         = config_flags.get('use_ff', False)
    use_filter     = config_flags.get('use_filter', False)
    alpha_ff       = config_flags.get('alpha_ff', 0.2)
    use_preview    = config_flags.get('use_preview', False)
    pose_ctrl_type = config_flags.get('pose_ctrl_type', 'pid')
    vel_ctrl_type  = config_flags.get('vel_ctrl_type', 'mpc')

    sys = lambda x, u: f_dyn(x, u, auv_params)

    # 2. Initial State Setup
    if DEBUG_MODE:
        x_curr = np.concatenate((np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0]), np.zeros(6)))
    else:
        x_curr = np.concatenate((eta_ref_full[0,:].flatten(), np.zeros(6)))

    eta_ref_prev = eta_ref_full[0]
    eta_dot_ref_filtered = np.zeros(6)

    # Initial Log
    history['t'].append(0.0)
    history['x'].append(x_curr.copy())
    history['tau'].append(np.zeros(6))
    history['nu_cmd_b'].append(np.zeros(6))

    # 3. Simulation Loop
    for i in range(num_steps):
        eta, nu = x_curr[0:6], x_curr[6:12]

        end_idx = min(i + N_horizon, len(eta_ref_full))
        eta_ref_window = eta_ref_full[i : end_idx, :]
        
        if len(eta_ref_window) < N_horizon:
            padding = np.repeat([eta_ref_window[-1]], N_horizon - len(eta_ref_window), axis=0)
            eta_ref_window = np.vstack((eta_ref_window, padding))

        # --- A. Position Control (NED) ---
        if DEBUG_MODE:
            nu_cmd_b = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
        else:
            if use_ff:
                eta_ref_dot_raw = cal_eta_err_with_ssa(eta_ref_window[0], eta_ref_prev) / dt
                eta_dot_ref_filtered = (alpha_ff * eta_ref_dot_raw) + (1.0 - alpha_ff) * eta_dot_ref_filtered if use_filter else eta_ref_dot_raw
                eta_ref_prev = eta_ref_window[0]
                nu_cmd_b = pose_ctrl.compute_control(eta, eta_ref_window[0], dt, eta_dot_ref_filtered)
            else:
                nu_cmd_b = pose_ctrl.compute_control(eta, eta_ref_window[0], dt)

        # --- B. Transformations (NED -> NWU) ---
        nu_nwu = T_ned_nwu @ nu
        nu_cmd_b_nwu = T_ned_nwu @ nu_cmd_b

        # --- C. Velocity Control (NWU) ---
        if 'mpc' in vel_ctrl_type and not DEBUG_MODE and use_preview:
            if pose_ctrl_type == 'pid':
                nu_hat_ned = generate_virtual_reference_pid(eta_ref_window, eta, nu_cmd_b, dt, pose_ctrl, pose_ctrl) 
            else:
                nu_hat_ned = generate_virtual_reference_ff_pi(eta_ref_window, eta, nu_cmd_b, use_filter, eta_dot_ref_filtered, alpha_ff, N_horizon, dt, pose_ctrl, pose_ctrl, use_ff)
            
            nu_hat_nwu = (nu_hat_ned @ T_ned_nwu.T)
            tau_cmd_nwu = vel_ctrl.compute_control(nu_nwu, nu_hat_nwu)
        else:
            if 'mpc' in vel_ctrl_type:
                tau_cmd_nwu = vel_ctrl.compute_control(nu_nwu, nu_cmd_b_nwu)
            else:
                tau_cmd_nwu = vel_ctrl.compute_control(nu_nwu, nu_cmd_b_nwu, dt)

        # --- D. Transformation (NWU -> NED) & Actuation Limits ---
        tau_cmd = T_ned_nwu @ np.asarray(tau_cmd_nwu).flatten()
        tau_cmd = np.clip(tau_cmd, -200, 200)
        
        # --- E. Plant Dynamics Integration ---
        x_curr = rk4_step(sys, x_curr, tau_cmd, dt)
        x_curr[3:6] = (x_curr[3:6] + np.pi) % (2 * np.pi) - np.pi

        # --- F. Logging ---
        history['t'].append((i + 1) * dt)
        history['x'].append(x_curr.copy())
        history['tau'].append(tau_cmd.copy())
        history['nu_cmd_b'].append(nu_cmd_b.copy())

    return history

def generate_config_report(cfg, file_path, title="UUV Controller Configuration"):
    """
    Generate TXT report แยกตาม Kinematics (eta) และ Dynamics (nu)
    - Position : eta [x, y, z, phi, theta, psi]
    - Velocity : nu  [u, v, w, p, q, r]
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("="*85 + "\n")
        f.write(f" {title.upper()}\n")
        f.write("="*85 + "\n")
        f.write(f"Model Engine : {cfg.get('model_name')} (v{cfg.get('model_version')})\n")
        f.write("-" * 85 + "\n\n")

        # กำหนด State Mapping ที่ถูกต้องตามหลักฟิสิกส์
        dofs_eta = [
            'x (Surge)', 'y (Sway)', 'z (Heave)', 
            'phi (Roll)', 'theta (Pitch)', 'psi (Yaw)'
        ]
        dofs_nu = [
            'u (Surge)', 'v (Sway)', 'w (Heave)', 
            'p (Roll Rate)', 'q (Pitch Rate)', 'r (Yaw Rate)'
        ]

        for ctrl_key in ['position_controller', 'velocity_controller']:
            if ctrl_key not in cfg: continue
            
            ctrl = cfg[ctrl_key]
            params = ctrl.get('params', {})
            f.write(f"[{ctrl_key.upper()}]\n")
            f.write(f"Type: {ctrl.get('type', 'N/A').upper()}\n")
            
            list_keys = [k for k, v in params.items() if isinstance(v, list) and len(v) == 6]
            scalars = {k: v for k, v in params.items() if k not in list_keys}

            if scalars:
                f.write(f"Settings: {scalars}\n")

            if not list_keys:
                f.write("No 6-DOF parameters found.\n\n")
                continue

            # Switch Logic: เลือกสัญลักษณ์และตัวแปรให้ตรงกับ Controller
            if ctrl_key == 'position_controller':
                dof_labels = dofs_eta
                state_sym = 'DOF (η)'
            else:
                dof_labels = dofs_nu
                state_sym = 'DOF (ν)'

            # Header
            header = f"{state_sym:<16} | " + " | ".join([f"{k:<12}" for k in list_keys])
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            # วนลูป 6 แถวตาม State Mapping
            for i in range(6):
                row_str = f"{dof_labels[i]:<16} | "
                vals = [f"{params[k][i]:<12.4f}" for k in list_keys]
                f.write(row_str + " | ".join(vals) + "\n")
            
            f.write("\n" + "-" * 85 + "\n\n")

    print(f"INFO: Config report saved successfully to {file_path}")

def main():
    
    mlflow_uri = "https://mlflow.amarr.tan" 
    mlflow.set_tracking_uri(mlflow_uri)

    # Load AUV Dynamics Setup
    auv_params = load_params('/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_descriptions/robots/xplorer_mini_dynamic_parameters.yaml')

    # Journal (option 1)
    # study_cases = {
    #     "Case_1_Model_Accuracy_without_Preview": [
    #         {"label": "PID -- PID",  "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/dpid_gain.yaml"},
    #         {"label": "PIwFF -- Nominal Koopman-Based MPC without preview (DMDc)",  "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/dmdc/constrained/without_preview/ffpi_stdmpc_gain.yaml"},
    #         {"label": "PIwFF -- Nominal Koopman-Based MPC without preview (eDMDc)", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/edmdc/constrained/without_preview/ffpi_stdmpc_gain.yaml"},
    #         {"label": "PIwFF -- Offset-Free Koopman-Based MPC without preview (DMDc)", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/dmdc/constrained/without_preview/ffpi_intmpc_gain.yaml"},
    #         {"label": "PIwFF -- Offset-Free Koopman-Based MPC without preview (eDMDc)", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/edmdc/constrained/without_preview/ffpi_intmpc_gain.yaml"}

    #     ],
    #     "Case_2_Preview_Impact_Nominal_MPC": [
    #         {"label": "PID -- PID",  "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/dpid_gain.yaml"},
    #         {"label": "PIwFF -- Nominal Koopman-Based MPC without preview (DMDc)", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/dmdc/constrained/without_preview/ffpi_stdmpc_gain.yaml"},
    #         {"label": "PIwFF -- Nominal Koopman-Based MPC with preview (DMDc)", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/dmdc/constrained/with_preview/ffpi_stdmpc_gain.yaml"},
    #         {"label": "PIwFF -- Nominal Koopman-Based MPC without preview (eDMDc)", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/edmdc/constrained/without_preview/ffpi_stdmpc_gain.yaml"},
    #         {"label": "PIwFF -- Nominal Koopman-Based MPC with preview (eDMDc)", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/edmdc/constrained/with_preview/ffpi_stdmpc_gain.yaml"}
    #     ],
    #     "Case_3_Preview_Impact_Offset-Free_MPC": [
    #         {"label": "PID -- PID",  "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/dpid_gain.yaml"},
    #         {"label": "PIwFF -- Offset-Free Koopman-Based MPC without preview (DMDc)", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/dmdc/constrained/without_preview/ffpi_intmpc_gain.yaml"},
    #         {"label": "PIwFF -- Offset-Free Koopman-Based MPC with preview (DMDc)", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/dmdc/constrained/with_preview/ffpi_intmpc_gain.yaml"},
    #         {"label": "PIwFF -- Offset-Free Koopman-Based MPC without preview (eDMDc)", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/edmdc/constrained/without_preview/ffpi_intmpc_gain.yaml"},
    #         {"label": "PIwFF -- Offset-Free Koopman-Based MPC with preview (eDMDc)", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/edmdc/constrained/with_preview/ffpi_intmpc_gain.yaml"}
    #     ],
    #     "Case_4_Offset-Free_MPC": [
    #         {"label": "PID -- PID",  "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/dpid_gain.yaml"},
    #         {"label": "PIwFF -- Nominal Koopman-Based MPC with preview (DMDc) ", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/dmdc/constrained/with_preview/ffpi_stdmpc_gain.yaml"},
    #         {"label": "PIwFF -- Offset-Free Koopman-Based MPC with preview (DMDc)", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/dmdc/constrained/with_preview/ffpi_intmpc_gain.yaml"},
    #         {"label": "PIwFF -- Nominal Koopman-Based MPC with preview (eDMDc) ", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/edmdc/constrained/with_preview/ffpi_stdmpc_gain.yaml"},
    #         {"label": "PIwFF -- Offset-Free Koopman-Based MPC with preview (eDMDc)", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/edmdc/constrained/with_preview/ffpi_intmpc_gain.yaml"}
    #     ]
    # }

    # Journal (option 2)
    study_cases = {
        "Case_1_Model_Accuracy_without_Preview": [
            {"label": "PIwFF -- Nominal Koopman-Based MPC without preview (DMDc)",  "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/dmdc/constrained/without_preview/ffpi_stdmpc_gain.yaml"},
            {"label": "PIwFF -- Nominal Koopman-Based MPC without preview (eDMDc)", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/edmdc/constrained/without_preview/ffpi_stdmpc_gain.yaml"},
            {"label": "PIwFF -- Offset-Free Koopman-Based MPC without preview (DMDc)", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/dmdc/constrained/without_preview/ffpi_intmpc_gain.yaml"},
            {"label": "PIwFF -- Offset-Free Koopman-Based MPC without preview (eDMDc)", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/edmdc/constrained/without_preview/ffpi_intmpc_gain.yaml"}

        ],
        "Case_2_Preview_Impact_Nominal_MPC": [
            {"label": "PIwFF -- Nominal Koopman-Based MPC without preview (DMDc)", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/dmdc/constrained/without_preview/ffpi_stdmpc_gain.yaml"},
            {"label": "PIwFF -- Nominal Koopman-Based MPC with preview (DMDc)", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/dmdc/constrained/with_preview/ffpi_stdmpc_gain.yaml"},
            {"label": "PIwFF -- Nominal Koopman-Based MPC without preview (eDMDc)", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/edmdc/constrained/without_preview/ffpi_stdmpc_gain.yaml"},
            {"label": "PIwFF -- Nominal Koopman-Based MPC with preview (eDMDc)", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/edmdc/constrained/with_preview/ffpi_stdmpc_gain.yaml"}
        ],
        "Case_3_Offset-Free_MPC": [
            {"label": "PID -- PID",  "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/dpid_gain.yaml"},
            {"label": "PIwFF -- Nominal Koopman-Based MPC with preview (DMDc) ", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/dmdc/constrained/with_preview/ffpi_stdmpc_gain.yaml"},
            {"label": "PIwFF -- Offset-Free Koopman-Based MPC with preview (DMDc)", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/dmdc/constrained/with_preview/ffpi_intmpc_gain.yaml"},
            {"label": "PIwFF -- Nominal Koopman-Based MPC with preview (eDMDc) ", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/edmdc/constrained/with_preview/ffpi_stdmpc_gain.yaml"},
            {"label": "PIwFF -- Offset-Free Koopman-Based MPC with preview (eDMDc)", "path": "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/gain/edmdc/constrained/with_preview/ffpi_intmpc_gain.yaml"}
        ]
    }

    # Define Base Output Directory
    base_result_dir = Path("/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/results/control")
    base_result_dir.mkdir(parents=True, exist_ok=True)
    mlflow_client = mlflow.MlflowClient()

    # Generate Trajectory
    dt = 0.1
    t_end = 60.0

    # eta_ref_full, _ = figure8_path(N=1, start_point=np.array([0,0,0.5,0,0,0]), end_point=np.array([0,0,0.5,0,0,0]), tfinal=t_end, dt=dt)
    eta_ref_full, _ = zigzag_path(num_zigs=2, start_point=np.array([0,0,0.2,0,0,0]), end_point=np.array([10,10,2.0,0,0,0]), dt=dt, tfinal=t_end)
    # eta_ref_full, _ = circle_path(N=2, init_pose=np.array([0,0,0.5,0,0,0]),desired_pose=np.array([10,10,5.0,0,0,0]), dt=dt, end_t=t_end)

    DEBUG_MODE = False
    hold_time = 40.0
    hold_steps = int(hold_time / dt)
    hold_trajectory = np.tile(eta_ref_full[-1, :], (hold_steps, 1)) 
    eta_ref_full = np.vstack((eta_ref_full, hold_trajectory))
    
    t_sim = t_end + hold_time

    # Batch Simulation Loop
    for idx_i, (study_name, configs) in enumerate(study_cases.items()):
        print(f"\n{'='*80}")
        print(f" Executing Study: {study_name}")
        print(f"{'='*80}")

        save_dir = base_result_dir / study_name
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / 'config').mkdir(parents=True, exist_ok=True)

        # Determine Max Horizon for current case
        parsed_configs = []
        for item in configs:
            with open(item["path"], 'r') as f:
                cfg = yaml.safe_load(f)['/**']['ros__parameters']
                txt_filename = save_dir / 'config' / f"{item['label']}.txt"
                generate_config_report(cfg, txt_filename, title=f"Report: {item['label']}")
                parsed_configs.append(cfg)

        histories = []
        labels = []

        # Run Simulations
        for idx_j, ros_params in enumerate(parsed_configs):
            label = configs[idx_j]["label"]
            labels.append(label)
            
            print(f"\n[{idx_j+1}/{len(configs)}] ========= Loading Model for {label} =========")
            print(f"Model Name: {ros_params['model_name']} | Version: {ros_params['model_version']}")
            koopman_model = load_model(client=mlflow_client, name=ros_params['model_name'], version=ros_params['model_version'])
            
            flags = get_config_flags(ros_params)
            node_name = f"sim_node_{idx_i}_{idx_j+1}"
            pose_ctrl, vel_ctrl = init_controllers(koopman_model, ros_params, node_name, sim_logger)

            print(f"Running Simulation: {label} ")
            history = run_cascade_simulation(auv_params, pose_ctrl, vel_ctrl, eta_ref_full, flags, t_sim, dt, DEBUG_MODE)
            histories.append(history)

        # Plot and Save Results
        print("\n========== Generating & Saving Plots ==========")
        plot_response(histories=histories, labels=labels, eta_ref_full=eta_ref_full, save_dir=save_dir)
       

if __name__ == "__main__":
    main()


    