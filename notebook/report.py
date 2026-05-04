import io
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from collections import defaultdict
from mcap_ros2.reader import read_ros2_messages
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. Helper Functions & Data Mapping
# ==========================================
def flatten_msg(msg_obj, prefix=""):
    items = {}
    for slot in dir(msg_obj):
        if slot.startswith('_') or callable(getattr(msg_obj, slot)): continue
        val = getattr(msg_obj, slot)
        key = f"{prefix}{slot}"
        if hasattr(val, '__slots__'): items.update(flatten_msg(val, prefix=f"{key}."))
        else: items[key] = val
    return items

def build_topic_map(df):
    ref_prefix, act_prefix, cmd_prefix = None, None, None
    for col in df.columns:
        if col.endswith('ref_nav.position.x') or col.endswith('ref_filtered.position.x'):
            ref_prefix = col.replace('.position.x', '')
        elif col.endswith('odom_filtered.pose.pose.position.x'):
            act_prefix = col.replace('.pose.pose.position.x', '')
        elif col.endswith('nu_cmd.linear.x'):
            cmd_prefix = col.replace('.linear.x', '')
            
    if not ref_prefix or not act_prefix: return None
    
    tmap = {
        'ref_pos_x': f'{ref_prefix}.position.x', 'ref_pos_y': f'{ref_prefix}.position.y', 'ref_pos_z': f'{ref_prefix}.position.z', 'ref_ori': f'{ref_prefix}.orientation',
        'act_pos_x': f'{act_prefix}.pose.pose.position.x', 'act_pos_y': f'{act_prefix}.pose.pose.position.y', 'act_pos_z': f'{act_prefix}.pose.pose.position.z', 'act_ori': f'{act_prefix}.pose.pose.orientation',
        'cmd_u': f'{cmd_prefix}.linear.x', 'cmd_v': f'{cmd_prefix}.linear.y', 'cmd_w': f'{cmd_prefix}.linear.z',
        'cmd_p': f'{cmd_prefix}.angular.x', 'cmd_q': f'{cmd_prefix}.angular.y', 'cmd_r': f'{cmd_prefix}.angular.z',
        'act_u': f'{act_prefix}.twist.twist.linear.x', 'act_v': f'{act_prefix}.twist.twist.linear.y', 'act_w': f'{act_prefix}.twist.twist.linear.z',
        'act_p': f'{act_prefix}.twist.twist.angular.x', 'act_q': f'{act_prefix}.twist.twist.angular.y', 'act_r': f'{act_prefix}.twist.twist.angular.z',
    }

    for col in df.columns:
        if col.endswith('est_tau.force.x'):
            tau_p = col.replace('.force.x', '')
            tmap.update({
                'tau_x': f'{tau_p}.force.x',  'tau_y': f'{tau_p}.force.y',  'tau_z': f'{tau_p}.force.z',
                'tau_k': f'{tau_p}.torque.x', 'tau_m': f'{tau_p}.torque.y', 'tau_n': f'{tau_p}.torque.z'
            })
            break
        elif col.endswith('wrench.force.x'):
            tau_p = col.replace('.wrench.force.x', '')
            tmap.update({
                'tau_x': f'{tau_p}.wrench.force.x',  'tau_y': f'{tau_p}.wrench.force.y',  'tau_z': f'{tau_p}.wrench.force.z',
                'tau_k': f'{tau_p}.wrench.torque.x', 'tau_m': f'{tau_p}.wrench.torque.y', 'tau_n': f'{tau_p}.wrench.torque.z'
            })
            break
            
    return tmap
# ==========================================
# 2. Data Loading & Statistical Filtering
# ==========================================
def load_and_sync_data(base_dir, controllers, path_type):
    base_path = Path(base_dir).expanduser().resolve()
    raw_storage = {}
    
    print("\n[*] Initializing Data Pipeline...")
    for ctrl in controllers:
        mcap_files = list((base_path / ctrl / path_type).rglob('*.mcap'))
        if not mcap_files: 
            print(f"    [!] Skip: {ctrl} (MCAP not found)")
            continue
        
        with open(mcap_files[0], 'rb') as f: mcap_stream = io.BytesIO(f.read())
            
        data_by_topic = defaultdict(list)
        for msg in read_ros2_messages(mcap_stream):
            topic = getattr(msg, 'topic', getattr(msg.channel, 'topic', 'unknown')).strip('/')
            flat = flatten_msg(msg.ros_msg); flat["_log_time"] = msg.log_time
            data_by_topic[topic].append(flat)
            
        dfs = [pd.DataFrame(recs).set_index('_log_time').rename(columns=lambda c: f"{t}.{c}") for t, recs in data_by_topic.items()]
        df = pd.concat(dfs, axis=1).sort_index()
        raw_storage[ctrl] = {'df': df, 'len': len(df)}
        print(f"    - {ctrl}: {len(df)} samples")

    if not raw_storage: return {}

    mean_len = np.mean([v['len'] for v in raw_storage.values()])
    filtered_storage = {k: v for k, v in raw_storage.items() if v['len'] >= mean_len * 0.5} # Tolerance added
    min_len = int(min(v['len'] for v in filtered_storage.values()))
    
    print(f"\n[*] Syncing Time Domain (Min Samples = {min_len})")
    
    comp_data = {}
    for ctrl, data in filtered_storage.items():
        df = data['df'].iloc[:min_len].copy()
        df = df.ffill().bfill().reset_index(names='_log_time')
        
        tmap = build_topic_map(df)
        if tmap:
            # Note: Converted to RADIANS to match the plotting backend requirements
            for ori in ['ref_ori', 'act_ori']:
                prefix = ori.split('_')[0]
                cols = [f"{tmap[ori]}.x", f"{tmap[ori]}.y", f"{tmap[ori]}.z", f"{tmap[ori]}.w"]
                if all(c in df.columns for c in cols):
                    e = R.from_quat(df[cols].to_numpy()).as_euler('xyz', degrees=False) 
                    df.loc[:, f'{prefix}_roll'] = e[:, 0]
                    df.loc[:, f'{prefix}_pitch'] = e[:, 1]
                    df.loc[:, f'{prefix}_yaw'] = e[:, 2]
            comp_data[ctrl] = {'df': df, 'tmap': tmap}
    return comp_data

# ==========================================
# 3. Enhanced Visualization & Metrics
# ==========================================
def plot_journal_style(comp_data, save_dir=None):
    labels = list(comp_data.keys())
    num_histories = len(labels)
    colors = cm.get_cmap('tab10')(np.linspace(0, 1, num_histories))
    
    # Extract Base Time
    base_ctrl = labels[0]
    df_b = comp_data[base_ctrl]['df']
    t_raw = df_b['_log_time'].values.astype(np.int64)
    t = (t_raw - t_raw[0]) / 1e9

    def get_arr(df, tmap, key):
        return df[tmap[key]].values if key in tmap and tmap[key] in df.columns else np.zeros(len(df))

    def get_arr_direct(df, key):
        return df[key].values if key in df.columns else np.zeros(len(df))

    def format_figure(fig, axs, bottom_margin):
        handles, labels_l = axs[0, 0].get_legend_handles_labels()
        fig.tight_layout(rect=[0, bottom_margin, 1, 0.96])
        fig.legend(handles, labels_l, loc='upper center', ncol=3, bbox_to_anchor=(0.5, bottom_margin), 
                   columnspacing=2.0, handletextpad=0.5, fontsize=9, frameon=True)

    # Extract Reference Arrays from Base Controller
    eta_ref = np.column_stack([get_arr(df_b, comp_data[base_ctrl]['tmap'], f'ref_pos_{x}') for x in ['x','y','z']] +
                              [get_arr_direct(df_b, f'ref_{a}') for a in ['roll','pitch','yaw']])

    # --- Fig 1: Position Tracking (Eta) ---
    fig1, axs1 = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    fig1.suptitle(r'Position Tracking ($\eta$)', fontsize=14, fontweight='bold')
    labels_eta_lin = ['x [m]', 'y [m]', 'z [m]']
    labels_eta_ang = [r'$\phi$ [deg]', r'$\theta$ [deg]', r'$\psi$ [deg]']

    for i in range(3):
        ref_ang_rad = np.unwrap(eta_ref[:, i+3]) if i == 2 else eta_ref[:, i+3]
        axs1[i, 0].plot(t, eta_ref[:, i], 'k--', linewidth=2, label='Ref' if i==0 else "")
        axs1[i, 1].plot(t, np.rad2deg(ref_ang_rad), 'k--', linewidth=2, label='Ref' if i==0 else "")
        
        for idx, ctrl in enumerate(labels):
            df, tmap = comp_data[ctrl]['df'], comp_data[ctrl]['tmap']
            pos = get_arr(df, tmap, f'act_pos_{["x","y","z"][i]}')
            ang_rad = get_arr_direct(df, f'act_{["roll","pitch","yaw"][i]}')
            state_ang_rad = np.unwrap(ang_rad) if i == 2 else ang_rad
            
            axs1[i, 0].plot(t, pos, color=colors[idx], label=ctrl if i==0 else "")
            axs1[i, 1].plot(t, np.rad2deg(state_ang_rad), color=colors[idx], label=ctrl if i==0 else "")
            
        axs1[i, 0].set_ylabel(labels_eta_lin[i]); axs1[i, 0].grid(True, linestyle=':', alpha=0.7)
        axs1[i, 1].set_ylabel(labels_eta_ang[i]); axs1[i, 1].grid(True, linestyle=':', alpha=0.7)
        
    axs1[2, 0].set_xlabel('Time [s]'); axs1[2, 1].set_xlabel('Time [s]')
    format_figure(fig1, axs1, bottom_margin=0.11)

    # --- Fig 2 & 3: Velocity Tracking & Error (Nu) ---
    fig2, axs2 = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    fig2.suptitle(r'Velocity Tracking ($\nu$)', fontsize=14, fontweight='bold')
    
    fig3, axs3 = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    fig3.suptitle(r'Velocity Tracking Error ($e_\nu$)', fontsize=14, fontweight='bold')
    
    labels_nu_lin, labels_nu_ang = ['u [m/s]', 'v [m/s]', 'w [m/s]'], ['p [deg/s]', 'q [deg/s]', 'r [deg/s]']
    labels_enu_lin, labels_enu_ang = [r'$e_u$ [m/s]', r'$e_v$ [m/s]', r'$e_w$ [m/s]'], [r'$e_p$ [deg/s]', r'$e_q$ [deg/s]', r'$e_r$ [deg/s]']

    lin_k, ang_k = ['u','v','w'], ['p','q','r']
    
    metrics_eta, metrics_nu = [], []

    for idx, ctrl in enumerate(labels):
        df, tmap = comp_data[ctrl]['df'], comp_data[ctrl]['tmap']
        
        eta = np.column_stack([get_arr(df, tmap, f'act_pos_{x}') for x in ['x','y','z']] +
                              [get_arr_direct(df, f'act_{a}') for a in ['roll','pitch','yaw']])
        nu = np.column_stack([get_arr(df, tmap, f'act_{k}') for k in lin_k] + [get_arr(df, tmap, f'act_{k}') for k in ang_k])
        nu_cmd = np.column_stack([get_arr(df, tmap, f'cmd_{k}') for k in lin_k] + [get_arr(df, tmap, f'cmd_{k}') for k in ang_k])

        # Calc Metrics
        e_lin = eta_ref[:, :3] - eta[:, :3]
        e_ang_rad = (eta_ref[:, 3:6] - eta[:, 3:6] + np.pi) % (2 * np.pi) - np.pi
        e_eta = np.hstack((e_lin, np.rad2deg(e_ang_rad)))
        metrics_eta.append({'rmse': np.sqrt(np.mean(e_eta**2, axis=0)), 'mae': np.mean(np.abs(e_eta), axis=0), 'maxae': np.max(np.abs(e_eta), axis=0)})
        
        e_nu = np.hstack((nu_cmd[:, :3] - nu[:, :3], np.rad2deg(nu_cmd[:, 3:6] - nu[:, 3:6])))
        metrics_nu.append({'rmse': np.sqrt(np.mean(e_nu**2, axis=0)), 'mae': np.mean(np.abs(e_nu), axis=0), 'maxae': np.max(np.abs(e_nu), axis=0)})

        for i in range(3):
            # Nu Plotting
            axs2[i, 0].plot(t, nu_cmd[:, i], '--', color=colors[idx], alpha=0.4, label=f'{ctrl} (Cmd)' if i==0 else "")
            axs2[i, 0].plot(t, nu[:, i], '-', color=colors[idx], label=ctrl if i==0 else "")
            axs2[i, 1].plot(t, np.rad2deg(nu_cmd[:, i+3]), '--', color=colors[idx], alpha=0.4)
            axs2[i, 1].plot(t, np.rad2deg(nu[:, i+3]), '-', color=colors[idx])
            
            # Nu Error Plotting
            axs3[i, 0].plot(t, nu_cmd[:, i] - nu[:, i], color=colors[idx], label=ctrl if i==0 else "")
            axs3[i, 1].plot(t, np.rad2deg(nu_cmd[:, i+3] - nu[:, i+3]), color=colors[idx], label=ctrl if i==0 else "")
            
    for i in range(3):
        axs2[i, 0].set_ylabel(labels_nu_lin[i]); axs2[i, 0].grid(True, linestyle=':', alpha=0.7)
        axs2[i, 1].set_ylabel(labels_nu_ang[i]); axs2[i, 1].grid(True, linestyle=':', alpha=0.7)
        axs3[i, 0].set_ylabel(labels_enu_lin[i]); axs3[i, 0].grid(True, linestyle=':', alpha=0.7)
        axs3[i, 1].set_ylabel(labels_enu_ang[i]); axs3[i, 1].grid(True, linestyle=':', alpha=0.7)

    axs2[2, 0].set_xlabel('Time [s]'); axs2[2, 1].set_xlabel('Time [s]')
    axs3[2, 0].set_xlabel('Time [s]'); axs3[2, 1].set_xlabel('Time [s]')
    format_figure(fig2, axs2, bottom_margin=0.18)
    format_figure(fig3, axs3, bottom_margin=0.13)

    # --- Fig 4: Generalized Torque (Tau) ---
    fig4, axs4 = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    fig4.suptitle(r'Generalized Torque ($\tau$)', fontsize=14, fontweight='bold')
    labels_tau_f, labels_tau_t = ['X [N]', 'Y [N]', 'Z [N]'], ['K [Nm]', 'M [Nm]', 'N [Nm]']

    tau_k_lin, tau_k_ang = ['x','y','z'], ['k','m','n']
    for idx, ctrl in enumerate(labels):
        df, tmap = comp_data[ctrl]['df'], comp_data[ctrl]['tmap']
        for i in range(3):
            axs4[i, 0].plot(t, get_arr(df, tmap, f'tau_{tau_k_lin[i]}'), color=colors[idx], label=ctrl if i==0 else "")
            axs4[i, 1].plot(t, get_arr(df, tmap, f'tau_{tau_k_ang[i]}'), color=colors[idx])

    for i in range(3):
        axs4[i, 0].set_ylabel(labels_tau_f[i]); axs4[i, 0].grid(True, linestyle=':', alpha=0.7)
        axs4[i, 1].set_ylabel(labels_tau_t[i]); axs4[i, 1].grid(True, linestyle=':', alpha=0.7)
        
    axs4[2, 0].set_xlabel('Time [s]'); axs4[2, 1].set_xlabel('Time [s]')
    format_figure(fig4, axs4, bottom_margin=0.11)

    # --- Fig 5: 3D Path Tracking ---
    fig5 = plt.figure(figsize=(10, 10))
    ax5 = fig5.add_subplot(111, projection='3d')
    ax5.plot(eta_ref[:, 0], eta_ref[:, 1], eta_ref[:, 2], 'k--', linewidth=2, label='Reference')
    
    for idx, ctrl in enumerate(labels):
        df, tmap = comp_data[ctrl]['df'], comp_data[ctrl]['tmap']
        ax5.plot(get_arr(df, tmap, 'act_pos_x'), get_arr(df, tmap, 'act_pos_y'), get_arr(df, tmap, 'act_pos_z'), 
                 color=colors[idx], label=ctrl)
        
    ax5.set_xlabel('x [m]'); ax5.set_ylabel('y [m]'); ax5.set_zlabel('z [m]')
    ax5.set_title('3D Path Tracking Comparison', fontsize=14, fontweight='bold', pad=20)
    handles5, labels5 = ax5.get_legend_handles_labels()
    fig5.tight_layout(rect=[0, 0.11, 1, 1])
    fig5.legend(handles5, labels5, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.11), columnspacing=2.0, fontsize=10, frameon=True)

    # --- Fig 6 & 7: Metric Tables ---
    def create_metric_figure(title_main, cols, metrics_data):
        fig, axs = plt.subplots(3, 1, figsize=(14, 8))
        fig.suptitle(title_main, fontweight='bold', fontsize=14)
        metric_keys, metric_titles = ['rmse', 'mae', 'maxae'], ['RMSE', 'MAE', 'MaxAE']
        colors_bg, highlight_color, highlight_text_color = ['#d9ead3', '#cfe2f3', '#f4cccc'], '#fff2cc', '#d62728'
        for i, key in enumerate(metric_keys):
            cell_text = [[f"{m[key][col_idx]:.4f}" for col_idx in range(6)] for m in metrics_data]
            tbl = axs[i].table(cellText=cell_text, rowLabels=labels, colLabels=cols, loc='center', cellLoc='center', 
                               rowColours=['#f2f2f2']*len(labels), colColours=[colors_bg[i]]*6)
            for col_idx in range(6):
                col_vals = [m[key][col_idx] for m in metrics_data]
                min_idx = np.argmin(col_vals)
                best_cell = tbl[min_idx + 1, col_idx]
                best_cell.set_facecolor(highlight_color)
                best_cell.get_text().set_weight('bold')
                best_cell.get_text().set_color(highlight_text_color)
            tbl.scale(1, 1.8); tbl.set_fontsize(11); axs[i].axis('off')
            axs[i].set_title(metric_titles[i], pad=5, fontweight='bold')
        fig.tight_layout()
        return fig

    fig6 = create_metric_figure(r'$\eta$ Error Metrics', ['x [m]', 'y [m]', 'z [m]', r'$\phi$ [deg]', r'$\theta$ [deg]', r'$\psi$ [deg]'], metrics_eta)
    fig7 = create_metric_figure(r'$\nu$ Error Metrics', ['u [m/s]', 'v [m/s]', 'w [m/s]', 'p [deg/s]', 'q [deg/s]', 'r [deg/s]'], metrics_nu)

    # --- File Export ---
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7]
        names = ["1_kinematics", "2_dynamics", "3_dynamics_error", "4_control_effort", "5_3d_path", "6_table_kinematics_metrics", "7_table_dynamics_metrics"]
        for f, name in zip(figs, names):
            f.savefig(save_path / f"{name}.svg", format='svg', bbox_inches='tight')
        plt.close('all') 
    else: 
        plt.show()

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    BASE_DIR = "/home/tanbjs/Desktop/test_result/journal/pool2_report"
    POOL_DIR = Path(BASE_DIR).expanduser()
    RESULT_DIR = POOL_DIR / "result"
    
    CONTROLLERS = [
        "pid_pi",
        "ffpi_dmdc_constrained_intergral_mpc_with_preview",
        "ffpi_edmdc_constrained_intergral_mpc_with_preview"           
    ]
    TARGET_PATH = "figure8"

    comp_data = load_and_sync_data(POOL_DIR, CONTROLLERS, TARGET_PATH)
    
    if comp_data:
        print("\n[*] Generating Journal Style Plots & Metrics...")
        plot_journal_style(comp_data, save_dir=RESULT_DIR)
        print(f"\n[Success] All graphs & tables exported to: {RESULT_DIR}")