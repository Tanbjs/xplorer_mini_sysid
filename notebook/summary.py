import argparse
import os
import io
import json
import traceback
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mcap_ros2.reader import read_ros2_messages
from scipy.spatial.transform import Rotation as R

def flatten_msg(msg_obj, prefix=""):
    items = {}
    for slot in dir(msg_obj):
        if slot.startswith('_') or callable(getattr(msg_obj, slot)): continue
        val = getattr(msg_obj, slot)
        key = f"{prefix}{slot}"
        if hasattr(val, '__slots__'):
            items.update(flatten_msg(val, prefix=f"{key}."))
        else: items[key] = val
    return items

def quat_to_euler(df, prefix):
    cols = [f'{prefix}.x', f'{prefix}.y', f'{prefix}.z', f'{prefix}.w']
    if all(c in df.columns for c in cols):
        quats = df[cols].astype(float).values
        mask = ~np.isnan(quats).any(axis=1)
        euler = np.full((len(df), 3), np.nan)
        if mask.any():
            euler[mask] = R.from_quat(quats[mask]).as_euler('xyz', degrees=True)
        df[f'{prefix}_roll'], df[f'{prefix}_pitch'], df[f'{prefix}_yaw'] = euler[:,0], euler[:,1], euler[:,2]
    return df

def wrap_angle_deg(angle_series):
    return (angle_series + 180.0) % 360.0 - 180.0

def calc_all_metrics(e_series, t_arr):
    e = np.asarray(e_series, dtype=float)
    t = np.asarray(t_arr, dtype=float)
    mask = ~np.isnan(e) & ~np.isnan(t)
    e_val, t_val = e[mask], t[mask]
    if len(e_val) == 0: return [np.nan]*6
    mse = np.mean(e_val**2); rmse = np.sqrt(mse)
    maxae = np.max(np.abs(e_val)); mae = np.mean(np.abs(e_val))
    ise = np.trapz(e_val**2, t_val); iae = np.trapz(np.abs(e_val), t_val)
    return [mse, rmse, maxae, mae, ise, iae]

def save_table_as_svg(df, title, filepath):
    fig, ax = plt.subplots(figsize=(12, df.shape[0] * 0.7 + 1.5))
    ax.axis('tight'); ax.axis('off')
    formatted_vals = [[f"{val:.4f}" if isinstance(val, (int, float)) else val for val in row] for row in df.values]
    table = ax.table(cellText=formatted_vals, rowLabels=df.index, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 2.5) 
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(weight='bold', color='white'); cell.set_facecolor('#2c3e50')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout(); plt.savefig(filepath, format='svg', bbox_inches='tight'); plt.close()

def generate_reports(df, target_dir, filename_base):
    
    # ==========================================
    # 1. AUTO-DETECT TOPIC PREFIXES
    # ==========================================
    ref_prefix = None
    for col in df.columns:
        if col.endswith('ref_nav.position.x'):
            ref_prefix = col.replace('.position.x', '')
            break
    if not ref_prefix:
        for col in df.columns:
            if col.endswith('ref_filtered.position.x'):
                ref_prefix = col.replace('.position.x', '')
                break
                
    act_prefix = None
    for col in df.columns:
        if col.endswith('odom_filtered.pose.pose.position.x'):
            act_prefix = col.replace('.pose.pose.position.x', '')
            break
            
    cmd_prefix = None
    for col in df.columns:
        if col.endswith('nu_cmd.linear.x'):
            cmd_prefix = col.replace('.linear.x', '')
            break

    if not ref_prefix or not act_prefix:
        print("\n[!] FATAL ERROR: Cannot detect Reference or Actual topics in the data.")
        sys.exit(1)

    print(f"   [-] Resolved Ref Prefix: {ref_prefix}")
    print(f"   [-] Resolved Act Prefix: {act_prefix}")

    # ==========================================
    # 2. DYNAMIC COLUMN MAPPING
    # ==========================================
    TOPIC_MAP = {
        'ref_pos_x': f'{ref_prefix}.position.x',
        'ref_pos_y': f'{ref_prefix}.position.y',
        'ref_pos_z': f'{ref_prefix}.position.z',
        'ref_ori':   f'{ref_prefix}.orientation',
        
        'act_pos_x': f'{act_prefix}.pose.pose.position.x',
        'act_pos_y': f'{act_prefix}.pose.pose.position.y',
        'act_pos_z': f'{act_prefix}.pose.pose.position.z',
        'act_ori':   f'{act_prefix}.pose.pose.orientation',
        
        'cmd_u': f'{cmd_prefix}.linear.x' if cmd_prefix else 'MISSING',
        'cmd_v': f'{cmd_prefix}.linear.y' if cmd_prefix else 'MISSING',
        'cmd_w': f'{cmd_prefix}.linear.z' if cmd_prefix else 'MISSING',
        'cmd_p': f'{cmd_prefix}.angular.x' if cmd_prefix else 'MISSING',
        'cmd_q': f'{cmd_prefix}.angular.y' if cmd_prefix else 'MISSING',
        'cmd_r': f'{cmd_prefix}.angular.z' if cmd_prefix else 'MISSING',
        
        'act_u': f'{act_prefix}.twist.twist.linear.x',
        'act_v': f'{act_prefix}.twist.twist.linear.y',
        'act_w': f'{act_prefix}.twist.twist.linear.z',
        'act_p': f'{act_prefix}.twist.twist.angular.x',
        'act_q': f'{act_prefix}.twist.twist.angular.y',
        'act_r': f'{act_prefix}.twist.twist.angular.z',
    }

    # ==========================================
    # 3. DATA PREPARATION
    # ==========================================
    time_raw = pd.to_numeric(df['_log_time'])
    t = ((time_raw - time_raw.iloc[0]) / 1e9).astype(float).values
    
    df = quat_to_euler(df, TOPIC_MAP['ref_ori'])
    df = quat_to_euler(df, TOPIC_MAP['act_ori'])
    
    ref_roll, ref_pitch, ref_yaw = f"{TOPIC_MAP['ref_ori']}_roll", f"{TOPIC_MAP['ref_ori']}_pitch", f"{TOPIC_MAP['ref_ori']}_yaw"
    act_roll, act_pitch, act_yaw = f"{TOPIC_MAP['act_ori']}_roll", f"{TOPIC_MAP['act_ori']}_pitch", f"{TOPIC_MAP['act_ori']}_yaw"

    ang_vel_cols = [TOPIC_MAP['cmd_p'], TOPIC_MAP['cmd_q'], TOPIC_MAP['cmd_r'], 
                    TOPIC_MAP['act_p'], TOPIC_MAP['act_q'], TOPIC_MAP['act_r']]
    for col in ang_vel_cols:
        if col in df.columns: df[col] = df[col].astype(float) * (180.0 / np.pi)

    pi_cols = ["MSE", "RMSE", "MaxAE", "MAE", "ISE", "IAE"]

    # ==========================================
    # 4. PLOTTING & REPORTING
    # ==========================================
    
    # --- 1. 3D Trajectory Plot ---
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(df[TOPIC_MAP['ref_pos_x']].astype(float), df[TOPIC_MAP['ref_pos_y']].astype(float), df[TOPIC_MAP['ref_pos_z']].astype(float), 'r--', label='Reference', alpha=0.7)
        ax.plot(df[TOPIC_MAP['act_pos_x']].astype(float), df[TOPIC_MAP['act_pos_y']].astype(float), df[TOPIC_MAP['act_pos_z']].astype(float), 'b-', label='Actual', alpha=0.8)
        ax.set_title("3D Path Tracking Trajectory", fontweight='bold')
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
        ax.legend(); ax.grid(True)
        plt.savefig(os.path.join(target_dir, f"3D_{filename_base}.svg"), format='svg')
        plt.close()
    except Exception as e: print(f"   [!] 3D Plot Error: {e}")

    # --- 2. XY Plot ---
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(df[TOPIC_MAP['ref_pos_x']].astype(float), df[TOPIC_MAP['ref_pos_y']].astype(float), 'r--', label='Reference')
        ax.plot(df[TOPIC_MAP['act_pos_x']].astype(float), df[TOPIC_MAP['act_pos_y']].astype(float), 'b-', label='Actual')
        ax.set_title("2D Trajectory Plot", fontweight='bold'); ax.legend(); ax.grid(True); ax.axis('equal')
        plt.savefig(os.path.join(target_dir, f"2D_{filename_base}.svg"), format='svg'); plt.close()
    except Exception as e: print(f"   [!] XY Error: {e}")

    # --- 3. ETA Tracking ---
    try:
        if act_yaw in df.columns:
            fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
            fig.suptitle("6-DOF Position & Orientation Tracking", fontsize=18, fontweight='bold')
            refs = [TOPIC_MAP['ref_pos_x'], ref_roll, TOPIC_MAP['ref_pos_y'], ref_pitch, TOPIC_MAP['ref_pos_z'], ref_yaw]
            acts = [TOPIC_MAP['act_pos_x'], act_roll, TOPIC_MAP['act_pos_y'], act_pitch, TOPIC_MAP['act_pos_z'], act_yaw]
            titles = ['X Position [m]', 'Roll [deg]', 'Y Position [m]', 'Pitch [deg]', 'Z Position [m]', 'Yaw [deg]']
            for i, ax in enumerate(axes.flat):
                if refs[i] in df.columns: ax.plot(t, df[refs[i]].astype(float), 'r--', label='Reference')
                if acts[i] in df.columns: ax.plot(t, df[acts[i]].astype(float), 'b-', label='Actual')
                ax.set_title(titles[i], fontweight='bold'); ax.grid(True, ls=':'); ax.legend(loc='upper right')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(target_dir, f"ETA_{filename_base}.svg"), format='svg'); plt.close()
    except Exception as e: print(f"   [!] ETA Tracking Error: {e}")

    # --- 4. ETA Errors & Tables ---
    try:
        if ref_yaw in df.columns and act_yaw in df.columns:
            err_eta = {
                'X [m]': df[TOPIC_MAP['ref_pos_x']].astype(float) - df[TOPIC_MAP['act_pos_x']].astype(float),
                'Y [m]': df[TOPIC_MAP['ref_pos_y']].astype(float) - df[TOPIC_MAP['act_pos_y']].astype(float),
                'Z [m]': df[TOPIC_MAP['ref_pos_z']].astype(float) - df[TOPIC_MAP['act_pos_z']].astype(float),
                'Roll [deg]': wrap_angle_deg(df[ref_roll].astype(float) - df[act_roll].astype(float)),
                'Pitch [deg]': wrap_angle_deg(df[ref_pitch].astype(float) - df[act_pitch].astype(float)),
                'Yaw [deg]': wrap_angle_deg(df[ref_yaw].astype(float) - df[act_yaw].astype(float))
            }
            fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
            fig.suptitle("6-DOF Position & Orientation Errors", fontsize=18, fontweight='bold')
            keys = list(err_eta.keys())
            for i, ax in enumerate(axes.flat):
                ax.plot(t, err_eta[keys[i]], color='blue')
                ax.set_title(f"Error: {keys[i]}", fontweight='bold'); ax.grid(True, ls=':')
                if i >= 4: ax.set_xlabel('Time [s]')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(target_dir, f"ERR_ETA_{filename_base}.svg"), format='svg'); plt.close()
            
            eta_res = {k: calc_all_metrics(v, t) for k, v in err_eta.items()}
            eta_df = pd.DataFrame.from_dict(eta_res, orient='index', columns=pi_cols)
            eta_df.to_json(os.path.join(target_dir, f"PI_ETA_{filename_base}.json"), orient='index', indent=4)
            save_table_as_svg(eta_df, "ETA Performance Index", os.path.join(target_dir, f"PI_ETA_{filename_base}.svg"))
    except Exception as e: print(f"   [!] ERR_ETA Error: {e}")

    # --- 5. NU Tracking & Errors ---
    try:
        if TOPIC_MAP['cmd_u'] in df.columns:
            fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
            fig.suptitle("6-DOF Velocity Tracking", fontsize=18, fontweight='bold')
            cmds = [TOPIC_MAP['cmd_u'], TOPIC_MAP['cmd_p'], TOPIC_MAP['cmd_v'], TOPIC_MAP['cmd_q'], TOPIC_MAP['cmd_w'], TOPIC_MAP['cmd_r']]
            acts = [TOPIC_MAP['act_u'], TOPIC_MAP['act_p'], TOPIC_MAP['act_v'], TOPIC_MAP['act_q'], TOPIC_MAP['act_w'], TOPIC_MAP['act_r']]
            titles = ['Surge (u) [m/s]', 'Roll Rate (p) [deg/s]', 'Sway (v) [m/s]', 'Pitch Rate (q) [deg/s]', 'Heave (w) [m/s]', 'Yaw Rate (r) [deg/s]']
            for i, ax in enumerate(axes.flat):
                ax.plot(t, df[cmds[i]].astype(float), 'r--', label='Reference')
                ax.plot(t, df[acts[i]].astype(float), 'b-', label='Actual')
                ax.set_title(titles[i], fontweight='bold'); ax.grid(True, ls=':'); ax.legend(loc='upper right')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(target_dir, f"NU_{filename_base}.svg"), format='svg'); plt.close()
            
            err_nu = {
                'u [m/s]': df[TOPIC_MAP['cmd_u']].astype(float) - df[TOPIC_MAP['act_u']].astype(float),
                'v [m/s]': df[TOPIC_MAP['cmd_v']].astype(float) - df[TOPIC_MAP['act_v']].astype(float),
                'w [m/s]': df[TOPIC_MAP['cmd_w']].astype(float) - df[TOPIC_MAP['act_w']].astype(float),
                'p [deg/s]': df[TOPIC_MAP['cmd_p']].astype(float) - df[TOPIC_MAP['act_p']].astype(float),
                'q [deg/s]': df[TOPIC_MAP['cmd_q']].astype(float) - df[TOPIC_MAP['act_q']].astype(float),
                'r [deg/s]': df[TOPIC_MAP['cmd_r']].astype(float) - df[TOPIC_MAP['act_r']].astype(float)
            }
            fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
            fig.suptitle("6-DOF Velocity Errors", fontsize=18, fontweight='bold')
            knu = list(err_nu.keys())
            for i, ax in enumerate(axes.flat):
                ax.plot(t, err_nu[knu[i]], color='blue')
                ax.set_title(f"Error: {knu[i]}", fontweight='bold'); ax.grid(True, ls=':')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(target_dir, f"ERR_NU_{filename_base}.svg"), format='svg'); plt.close()
            
            nu_res = {k: calc_all_metrics(v, t) for k, v in err_nu.items()}
            nu_df = pd.DataFrame.from_dict(nu_res, orient='index', columns=pi_cols)
            nu_df.to_json(os.path.join(target_dir, f"PI_NU_{filename_base}.json"), orient='index', indent=4)
            save_table_as_svg(nu_df, "NU Performance Index", os.path.join(target_dir, f"PI_NU_{filename_base}.svg"))
    except Exception as e: print(f"   [!] NU Tracking/Error Failed: {e}")

def process_file(mcap_path):
    print(f"Processing: {os.path.basename(mcap_path)}")
    base_dir = os.path.dirname(os.path.abspath(mcap_path))
    fn_base = os.path.basename(mcap_path).replace(".mcap", "")
    
    try:
        with open(mcap_path, 'rb') as f:
            mcap_stream = io.BytesIO(f.read())
        
        data_by_topic = {}
        for msg in read_ros2_messages(mcap_stream):
            topic = getattr(msg, 'topic', getattr(msg.channel, 'topic', 'unknown')).strip('/')
            flat = flatten_msg(msg.ros_msg)
            flat["_log_time"] = msg.log_time
            if topic not in data_by_topic: data_by_topic[topic] = []
            data_by_topic[topic].append(flat)
            
        if not data_by_topic:
            print("   [!] No data found in MCAP.")
            return

        dfs = []
        for topic, records in data_by_topic.items():
            df_t = pd.DataFrame(records)
            df_t.set_index('_log_time', inplace=True)
            df_t.columns = [f"{topic}.{c}" for c in df_t.columns]
            dfs.append(df_t)
            
        df_merged = pd.concat(dfs, axis=1).sort_index().ffill().dropna(how='all')
        df_merged['_log_time'] = df_merged.index
        df_merged.reset_index(drop=True, inplace=True)

        print(f"   [-] Synchronized Data Shape: {df_merged.shape}")
        
        generate_reports(df_merged, base_dir, fn_base)
        print(f"Success! All Reports saved in {base_dir}")
        
    except SystemExit:
        pass
    except Exception as e: 
        print(f"Error processing file: {e}")
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    if os.path.isdir(args.path):
        for f in os.listdir(args.path):
            if f.endswith(".mcap"): process_file(os.path.join(args.path, f))
    else: process_file(args.path)

if __name__ == "__main__": 
    main()