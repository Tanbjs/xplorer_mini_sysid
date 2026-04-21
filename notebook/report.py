import io
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from mcap_ros2.reader import read_ros2_messages
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. Helper Functions
# ==========================================
def wrap_angle_deg(angle_array):
    return (angle_array + 180.0) % 360.0 - 180.0

def flatten_msg(msg_obj, prefix=""):
    items = {}
    for slot in dir(msg_obj):
        if slot.startswith('_') or callable(getattr(msg_obj, slot)): continue
        val = getattr(msg_obj, slot)
        key = f"{prefix}{slot}"
        if hasattr(val, '__slots__'): items.update(flatten_msg(val, prefix=f"{key}."))
        else: items[key] = val
    return items

# ==========================================
# 2. Dynamic Mapping System
# ==========================================
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
    return {
        'ref_pos_x': f'{ref_prefix}.position.x', 'ref_pos_y': f'{ref_prefix}.position.y', 'ref_pos_z': f'{ref_prefix}.position.z', 'ref_ori': f'{ref_prefix}.orientation',
        'act_pos_x': f'{act_prefix}.pose.pose.position.x', 'act_pos_y': f'{act_prefix}.pose.pose.position.y', 'act_pos_z': f'{act_prefix}.pose.pose.position.z', 'act_ori': f'{act_prefix}.pose.pose.orientation',
        'cmd_u': f'{cmd_prefix}.linear.x', 'cmd_v': f'{cmd_prefix}.linear.y', 'cmd_w': f'{cmd_prefix}.linear.z',
        'cmd_p': f'{cmd_prefix}.angular.x', 'cmd_q': f'{cmd_prefix}.angular.y', 'cmd_r': f'{cmd_prefix}.angular.z',
        'act_u': f'{act_prefix}.twist.twist.linear.x', 'act_v': f'{act_prefix}.twist.twist.linear.y', 'act_w': f'{act_prefix}.twist.twist.linear.z',
        'act_p': f'{act_prefix}.twist.twist.angular.x', 'act_q': f'{act_prefix}.twist.twist.angular.y', 'act_r': f'{act_prefix}.twist.twist.angular.z',
    }

# ==========================================
# 3. Data Loading & Statistical Filtering
# ==========================================
def load_and_sync_data(base_dir, controllers, path_type):
    base_path = Path(base_dir).expanduser().resolve()
    raw_storage = {}
    
    print("\n[*] Step 1: Loading MCAP files and performing Initial Sample Count...")
    for ctrl in controllers:
        mcap_files = list((base_path / ctrl / path_type).rglob('*.mcap'))
        if not mcap_files: 
            print(f"    [!] Skip: {ctrl} (MCAP not found)")
            continue
        
        print(f"    - Loading {ctrl}...")
        with open(mcap_files[0], 'rb') as f:
            mcap_stream = io.BytesIO(f.read())
            
        data_by_topic = defaultdict(list)
        for msg in read_ros2_messages(mcap_stream):
            topic = getattr(msg, 'topic', getattr(msg.channel, 'topic', 'unknown')).strip('/')
            flat = flatten_msg(msg.ros_msg); flat["_log_time"] = msg.log_time
            data_by_topic[topic].append(flat)
            
        dfs = [pd.DataFrame(recs).set_index('_log_time').rename(columns=lambda c: f"{t}.{c}") for t, recs in data_by_topic.items()]
        df = pd.concat(dfs, axis=1).sort_index()
        raw_storage[ctrl] = {'df': df, 'len': len(df)}
        print(f"      -> Total Samples: {len(df)}")

    if not raw_storage: return {}

    all_lengths = [v['len'] for v in raw_storage.values()]
    mean_len = np.mean(all_lengths)
    print(f"\n[*] Step 2: Statistical Outlier Detection (Mean = {mean_len:.2f})")
    
    filtered_storage = {}
    for ctrl, data in raw_storage.items():
        if data['len'] < mean_len:
            print(f"    [REJECTED] {ctrl}: {data['len']} pts < Mean. Data likely corrupted.")
        else:
            print(f"    [PASSED]   {ctrl}: {data['len']} pts")
            filtered_storage[ctrl] = data

    if not filtered_storage: return {}

    min_len = int(min(v['len'] for v in filtered_storage.values()))
    print(f"\n[*] Step 3: Time Trimming (Syncing to Global Min = {min_len} samples)")
    
    comp_data = {}
    for ctrl, data in filtered_storage.items():
        diff = data['len'] - min_len
        if diff > 0:
            print(f"    [-] {ctrl}: Clipping last {diff} samples.")
            
        df = data['df'].iloc[:min_len].copy()
        df = df.ffill().bfill().reset_index(names='_log_time')
        
        tmap = build_topic_map(df)
        if tmap:
            for ori in ['ref_ori', 'act_ori']:
                prefix = ori.split('_')[0]
                cols = [f"{tmap[ori]}.x", f"{tmap[ori]}.y", f"{tmap[ori]}.z", f"{tmap[ori]}.w"]
                if all(c in df.columns for c in cols):
                    e = R.from_quat(df[cols].to_numpy()).as_euler('xyz', degrees=True)
                    df.loc[:, f'{prefix}_roll'] = e[:, 0]
                    df.loc[:, f'{prefix}_pitch'] = e[:, 1]
                    df.loc[:, f'{prefix}_yaw'] = e[:, 2]
            comp_data[ctrl] = {'df': df, 'tmap': tmap}
    return comp_data

# ==========================================
# 4. Evaluation & Plotting
# ==========================================
ETA_GRID = [[('X (m)', 'act_pos_x', 'ref_pos_x'), ('Phi (deg)', 'act_roll', 'ref_roll')],
            [('Y (m)', 'act_pos_y', 'ref_pos_y'), ('Theta (deg)', 'act_pitch', 'ref_pitch')],
            [('Z (m)', 'act_pos_z', 'ref_pos_z'), ('Psi (deg)', 'act_yaw', 'ref_yaw')]]

def calculate_metrics(data_dict, ctrl_name):
    df, tmap = data_dict['df'], data_dict['tmap']
    metrics = {'Controller': ctrl_name}
    NU_GRID = [[('U (m/s)', 'act_u', 'cmd_u'), ('P (rad/s)', 'act_p', 'cmd_p')],
               [('V (m/s)', 'act_v', 'cmd_v'), ('Q (rad/s)', 'act_q', 'cmd_q')],
               [('W (m/s)', 'act_w', 'cmd_w'), ('R (rad/s)', 'act_r', 'cmd_r')]]
    grid = [i for row in ETA_GRID+NU_GRID for i in row]
    for name, ak, rk in grid:
        # Resolve Columns Strictly
        if any(x in ak for x in ['roll', 'pitch', 'yaw']):
            ac, rc = f"act_{ak.split('_')[-1]}", f"ref_{ak.split('_')[-1]}"
        else:
            ac, rc = tmap.get(ak), tmap.get(rk)
            
        if ac in df.columns and rc in df.columns:
            err = df[rc].to_numpy() - df[ac].to_numpy()
            if 'deg' in name: err = wrap_angle_deg(err)
            metrics[f'RMSE_{name}'] = np.sqrt(np.mean(err**2))
            metrics[f'MAE_{name}'] = np.mean(np.abs(err))
            metrics[f'MaxAE_{name}'] = np.max(np.abs(err))
    return metrics

def plot_eta_grid(comp_data, save_dir):
    # ขนาดรูปภาพ 10x8
    fig_resp, ax_resp = plt.subplots(3, 2, figsize=(10, 8))
    fig_err, ax_err = plt.subplots(3, 2, figsize=(10, 8))
    
    ctrl_list = list(comp_data.keys())

    # บังคับชื่อแกนเป็น Greek + Degree Symbol (LaTeX)
    labels = [
        ['X [m]', r'$\phi$ [$^\circ$]'],
        ['Y [m]', r'$\theta$ [$^\circ$]'],
        ['Z [m]', r'$\psi$ [$^\circ$]']
    ]

    for ctrl, data in comp_data.items():
        df, tmap = data['df'], data['tmap']
        t_raw = df['_log_time'].values.astype(np.int64)
        time = (t_raw - t_raw[0]) / 1e9
        
        for r in range(3):
            for c in range(2):
                orig_name, ak, rk = ETA_GRID[r][c]
                
                if any(x in ak for x in ['roll', 'pitch', 'yaw']):
                    ac, rc = f"act_{ak.split('_')[-1]}", f"ref_{ak.split('_')[-1]}"
                else:
                    ac, rc = tmap.get(ak), tmap.get(rk)
                
                if ac in df.columns:
                    ax_resp[r, c].plot(time, df[ac], label=ctrl, linewidth=1.2, zorder=2)
                    
                    if rc in df.columns:
                        err = df[rc].to_numpy() - df[ac].to_numpy()
                        is_deg = 'deg' in orig_name or any(x in ak for x in ['roll', 'pitch', 'yaw'])
                        ax_err[r, c].plot(time, wrap_angle_deg(err) if is_deg else err, 
                                          label=ctrl, linewidth=1.2, zorder=2)
                        
                        if ctrl == ctrl_list[-1]:
                            ax_resp[r, c].plot(time, df[rc], 'k--', label='reference', 
                                              alpha=0.9, linewidth=1.5, zorder=10)
                
                display_name = labels[r][c]
                ax_resp[r, c].set_title(display_name, fontsize=10, fontweight='bold')
                ax_err[r, c].set_title(f"{display_name} Error", fontsize=10, fontweight='bold')
                
                # Physical Scaling
                if 'Z [m]' in display_name:
                    ax_resp[r, c].set_ylim(-1.0, 0.0)
                elif any(sym in display_name for sym in [r'$\phi$', r'$\theta$']):
                    ax_resp[r, c].set_ylim(-10, 10)
                elif r'$\psi$' in display_name:
                    ax_resp[r, c].set_ylim(-200, 200)
                
                ax_resp[r, c].legend(fontsize='x-small', loc='upper right')
                ax_err[r, c].legend(fontsize='x-small', loc='upper right')
                ax_resp[r, c].grid(True, linestyle=':', alpha=0.6)
                ax_err[r, c].grid(True, linestyle=':', alpha=0.6)

    fig_resp.tight_layout(pad=1.5)
    fig_err.tight_layout(pad=1.5)
    fig_resp.savefig(save_dir / "eta_resp.svg")
    fig_err.savefig(save_dir / "eta_err.svg")
    plt.close('all')

def plot_2d_path(comp_data, save_dir):
    plt.figure(figsize=(10, 8))
    for ctrl, data in comp_data.items():
        df, tmap = data['df'], data['tmap']
        act_x, act_y = tmap.get('act_pos_x'), tmap.get('act_pos_y')
        ref_x, ref_y = tmap.get('ref_pos_x'), tmap.get('ref_pos_y')
        if act_x in df.columns and act_y in df.columns:
            plt.plot(df[act_x], df[act_y], label=ctrl)
            if ctrl == list(comp_data.keys())[0] and ref_x in df.columns:
                plt.plot(df[ref_x], df[ref_y], 'k--', label='Reference', alpha=0.6)
    plt.title('2D Path Response (X-Y)', fontweight='bold'); plt.xlabel('X (m)'); plt.ylabel('Y (m)')
    plt.legend(); plt.grid(True); plt.axis('equal'); plt.tight_layout()
    plt.savefig(save_dir / "2d_path_response.svg"); plt.close()

def save_table_as_svg(df, title, filename, output_dir):
    df_str = df.map(lambda x: "-" if pd.isna(x) else (f"{x:.4e}" if abs(x) < 0.001 and x != 0 else f"{x:.4f}"))
    fig, ax = plt.subplots(figsize=(max(12, len(df.columns)*2), 2+len(df)*0.8)); ax.axis('off')
    table = ax.table(cellText=df_str.values, rowLabels=df_str.index, colLabels=df_str.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.1, 2)
    for (r, c), cell in table.get_celld().items():
        if r == 0: cell.set_facecolor('#2C3E50'); cell.set_text_props(color='white', weight='bold')
        elif c == -1: cell.set_facecolor('#ECF0F1'); cell.set_text_props(weight='bold')
        elif r > 0 and c >= 0:
            if df.iloc[r-1, c] == df.iloc[:, c].min():
                cell.set_facecolor('#FADBD8'); cell.set_text_props(color='#C0392B', weight='bold')
    plt.title(title, weight='bold', pad=30); plt.savefig(output_dir / filename, bbox_inches='tight'); plt.close()

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    BASE_DIR = "~/Desktop/test_result/journal/pool"
    # BASE_DIR = "/home/tanbjs/Desktop/test_result/journal/gazebo/pose_gt/without_oc"
    POOL_DIR = Path(BASE_DIR).expanduser()
    RESULT_DIR = POOL_DIR / "result"; RESULT_DIR.mkdir(parents=True, exist_ok=True)
    
    CONTROLLERS = [
                "pid-pid", 
                "pi_dmdc_lqt_without_preview",
                "pi_dmdc_lqt_with_preview",
                "pi_dmdc_mpc_without_preview", 
                "pi_dmdc_mpc_with_preview", 
                "pi_edmdc_lqt_without_preview",
                "pi_edmdc_lqt_with_preview",
                "pi_edmdc_mpc_without_preview", 
                "pi_edmdc_mpc_with_preview"]
    TARGET_PATH = "figure8"

    comp_data = load_and_sync_data(POOL_DIR, CONTROLLERS, TARGET_PATH)
    
    if comp_data:
        print("\n[*] Step 4: Generating Comparison Plots (Size 10x8)...")
        plot_2d_path(comp_data, RESULT_DIR)
        plot_eta_grid(comp_data, RESULT_DIR)
        
        print("\n[*] Step 5: Calculating Metrics and Tables...")
        metrics = [calculate_metrics(d, c) for c, d in comp_data.items()]
        for m_type in ["RMSE", "MaxAE", "MAE"]:
            for group, cols in [("Eta", ['X (m)', 'Y (m)', 'Z (m)', 'Phi (deg)', 'Theta (deg)', 'Psi (deg)']), 
                               ("Nu", ['U (m/s)', 'V (m/s)', 'W (m/s)', 'P (rad/s)', 'Q (rad/s)', 'R (rad/s)'])]:
                table_data = {mt['Controller']: {c: mt.get(f'{m_type}_{c}', np.nan) for c in cols} for mt in metrics}
                df_table = pd.DataFrame.from_dict(table_data, orient='index').dropna(axis=1, how='all')
                if not df_table.empty:
                    save_table_as_svg(df_table, f"{m_type} - {group}", f"table_{m_type.lower()}_{group.lower()}.svg", RESULT_DIR)
        
        print(f"\n[Success] Processed. Results in: {RESULT_DIR}")