"""
Evaluation script for Koopman-based system identification models (DMDc / eDMDc).

Implements the multi-step prediction protocol described in the journal paper:
  - p-step ahead open-loop prediction with periodic re-initialization
  - Per-trajectory and ensemble-averaged RMSE
  - One-step and multi-step prediction visualization

Naming convention follows the paper: "DMDc" and "eDMDc" (lowercase 'e').
"""

import warnings
import os
import json
import logging
from pathlib import Path

import urllib3
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from minio import Minio
from minio.error import S3Error

warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
sim_logger = logging.getLogger("Simulation")
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

from xplorer_mini_sysid.lib.utils.mlflow import load_model
from kmc.utils.model_wrapper import DMDcWrapper, EDMDcWrapper, DeepModelWrapper


# =============================================================================
# Prediction routines
# =============================================================================

def _call_predict(model_wrapper, x_k, u_k):
    """
    Unified predict call. Returns (next_lifted_state, observation_at_k+1).

    The wrapper's predict() returns x_{k+1} (DMDc) or (z_{k+1}, y_{k+1}) (eDMDc),
    where y is the de-lifted observation. Both correspond to the NEXT step.
    """
    model_input = {'x': x_k, 'u': u_k}
    if isinstance(model_wrapper, DMDcWrapper):
        y_next = model_wrapper.predict(context=None, model_input=model_input)
        return y_next, y_next  # DMDc: next state == observation
    elif isinstance(model_wrapper, EDMDcWrapper):
        x_next, y_next = model_wrapper.predict(context=None, model_input=model_input)
        return x_next, y_next
    else:
        raise TypeError(f"Unsupported wrapper type: {type(model_wrapper)}")


def one_step_prediction(model_wrapper, x_curr, u_curr):
    """
    One-step-ahead prediction.

    For each k, computes hat{x}_{k+1} = f(x_k, u_k) and stores it at index k+1.
    The slot y_pred[0] is filled with x_curr[0] since no prediction exists for
    the initial step (zero contribution to RMSE).

    Args:
        model_wrapper: DMDcWrapper or EDMDcWrapper
        x_curr: ground-truth state trajectory, shape (N, n_x)
        u_curr: control input trajectory, shape (N_u, n_u) where N_u <= N

    Returns:
        y_pred: predictions aligned with x_curr, shape (N, n_x)
    """
    N = x_curr.shape[0]
    y_pred = np.zeros((N, x_curr.shape[1]))
    y_pred[0, :] = x_curr[0, :]  # k=0: no prediction; use ground truth

    for i in range(u_curr.shape[0]):
        _, y_next = _call_predict(model_wrapper, x_curr[i, :], u_curr[i, :])
        # predict(x_i, u_i) -> hat{x}_{i+1}: store at index i+1
        if i + 1 < N:
            y_pred[i + 1, :] = y_next.squeeze()

    return y_pred


def multi_step_prediction(model_wrapper, step, x_curr, u_curr):
    """
    p-step ahead open-loop prediction with periodic re-initialization.

    Implements equation (multi_step_prediction) in the paper:
      hat{z}_k = A^j z_ell + sum_{s=0}^{j-1} A^{j-1-s} B tau_{ell+s}
    where ell = floor(k/p) * p is the most recent re-init index and j = k - ell.

    At each re-initialization point k = ell, the lifted state is reset to the
    ground-truth observation and y_pred[ell] is set to x_curr[ell] (i.e., j=0).

    Args:
        model_wrapper: DMDcWrapper or EDMDcWrapper
        step: prediction horizon p
        x_curr: ground-truth state trajectory, shape (N, n_x)
        u_curr: control input trajectory, shape (N_u, n_u) where N_u <= N

    Returns:
        y_pred: predictions aligned with x_curr, shape (N, n_x)
    """
    N = x_curr.shape[0]
    y_pred = np.zeros((N, x_curr.shape[1]))
    y_pred[0, :] = x_curr[0, :]

    x_k = x_curr[0, :].copy()
    for i in range(u_curr.shape[0]):
        if i % step == 0:
            x_k = x_curr[i, :].copy()
            y_pred[i, :] = x_curr[i, :]   # j=0: re-init point uses ground truth

        next_state, y_next = _call_predict(model_wrapper, x_k, u_curr[i, :])

        # predict(x_k, u_i) -> hat{x}_{i+1}: store at index i+1
        if i + 1 < N:
            y_pred[i + 1, :] = y_next.squeeze()
        x_k = next_state.squeeze()

    return y_pred


# =============================================================================
# Metrics
# =============================================================================

def calculate_metrics(y_true, y_pred):
    """
    Compute per-channel prediction metrics.

    Returns:
        nrmse: NRMSE in MATLAB SystemID-Toolbox sense, ||e||_2 / ||y - mean(y)||_2
        nmse:  NRMSE squared
        rmse:  sqrt(mean((y - y_hat)^2))   <-- matches paper definition
        fit_pct: 100 * (1 - nrmse)
        r2_pct:  100 * (1 - nmse)
    """
    Ns = y_true.shape[0]
    err = y_true - y_pred

    norm_err = np.linalg.norm(err, axis=0)
    norm_ref_dev = np.linalg.norm(y_true - np.mean(y_true, axis=0), axis=0)
    norm_ref_dev = np.where(norm_ref_dev == 0, 1e-10, norm_ref_dev)

    mse = (norm_err**2) / Ns
    nrmse = norm_err / norm_ref_dev
    nmse = (norm_err**2) / (norm_ref_dev**2)
    rmse = np.sqrt(mse)

    fit_pct = 100.0 * (1.0 - nrmse)
    r2_pct = 100.0 * (1.0 - nmse)

    return nrmse, nmse, rmse, fit_pct, r2_pct


# =============================================================================
# Reporting
# =============================================================================

MODEL_NAMES = ['DMDc', 'eDMDc']  # paper convention: lowercase 'e'


def save_metric_tables_as_svg(dfs_dict, main_title, file_path):
    """Render a stack of metric tables and save as SVG."""
    fig, axes = plt.subplots(5, 1, figsize=(12, 14))
    fig.patch.set_facecolor('#ffffff')
    fig.suptitle(main_title, fontsize=18, weight='bold', y=0.98,
                 fontfamily='sans-serif', color='#1a1a1a')

    for ax, (metric_name, df) in zip(axes, dfs_dict.items()):
        ax.axis('off')
        ax.set_title(metric_name, fontsize=14, weight='bold',
                     fontfamily='sans-serif', pad=10, color='#2c3e50')

        df_disp = df.copy()
        for col in df_disp.columns:
            if col != 'Model':
                if metric_name in ['Fit (NRMSE) [%]', 'R² (%)']:
                    df_disp[col] = df_disp[col].apply(lambda x: f"{x:.2f}")
                else:
                    df_disp[col] = df_disp[col].apply(lambda x: f"{x:.4f}")

        table = ax.table(cellText=df_disp.values, colLabels=df_disp.columns,
                         loc='center', cellLoc='center', bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)

        is_higher_better = metric_name in ['Fit (NRMSE) [%]', 'R² (%)']

        # Pre-compute best value per data column (model-agnostic; works for >2 models)
        data_cols = [c for c in df.columns if c != 'Model']
        best_vals = {}
        for c in data_cols:
            vals = df[c].values
            best_vals[c] = (np.max(vals) if is_higher_better else np.min(vals)) \
                if not np.allclose(vals, vals[0]) else None

        for (row, col_idx), cell in table.get_celld().items():
            cell.set_edgecolor('#dddddd')
            cell.set_linewidth(0.5)

            if row == 0:
                cell.set_text_props(weight='bold', color='white', fontfamily='sans-serif')
                cell.set_facecolor('#1e3d59')
            else:
                cell.set_text_props(fontfamily='sans-serif', color='#333333')
                cell.set_facecolor('#f4f7f6' if row % 2 == 0 else '#ffffff')

                if col_idx > 0:
                    col_name = df.columns[col_idx]
                    current_val = df.iloc[row - 1][col_name]
                    best = best_vals.get(col_name)
                    if best is not None and current_val == best:
                        cell.set_text_props(weight='bold', color='#16a085')

    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.0)
    fig.savefig(file_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close(fig)
    sim_logger.info(f"Saved Metric Tables to {file_path}")


def _build_metric_dfs(dmdc_m, edmdc_m):
    """Build dict of DataFrames for each metric, one row per model."""
    metrics_names = ['RMSE', 'NMSE', 'NRMSE', 'Fit (NRMSE) [%]', 'R² (%)']
    metrics_idx = [2, 1, 0, 3, 4]
    dfs = {}
    for name, idx in zip(metrics_names, metrics_idx):
        data = {'Model': MODEL_NAMES}
        if name == 'RMSE':
            cols = ['u [m/s]', 'v [m/s]', 'w [m/s]',
                    'p [deg/s]', 'q [deg/s]', 'r [deg/s]']
        else:
            cols = ['u', 'v', 'w', 'p', 'q', 'r']
        for i, col in enumerate(cols):
            data[col] = [dmdc_m[idx][i], edmdc_m[idx][i]]
        dfs[name] = pd.DataFrame(data)
    return dfs


def report_metrics(results, save_dir: Path, step: int):
    """Compute, save, and aggregate per-trajectory metrics."""
    dmdc_one_list, edmdc_one_list = [], []
    dmdc_multi_list, edmdc_multi_list = [], []

    for result in results:
        traj = result['trajectory']
        traj_name = result['traj_name']
        true_state = traj[[c for c in traj.columns
                          if c.startswith('odom_filtered.twist.twist')]].values

        d_one = calculate_metrics(true_state, result['dmdc_one_step_pred'])
        e_one = calculate_metrics(true_state, result['edmdc_one_step_pred'])
        d_multi = calculate_metrics(true_state, result['dmdc_multi_step_pred'])
        e_multi = calculate_metrics(true_state, result['edmdc_multi_step_pred'])

        dmdc_one_list.append(d_one)
        edmdc_one_list.append(e_one)
        dmdc_multi_list.append(d_multi)
        edmdc_multi_list.append(e_multi)

        traj_dir = save_dir / traj_name
        traj_dir.mkdir(parents=True, exist_ok=True)

        save_metric_tables_as_svg(_build_metric_dfs(d_one, e_one),
                                  "One-Step Prediction Metrics",
                                  traj_dir / 'metrics_one_step.svg')
        save_metric_tables_as_svg(_build_metric_dfs(d_multi, e_multi),
                                  f"Multi-Step (p={step}) Prediction Metrics",
                                  traj_dir / 'metrics_multi_step.svg')

    def avg_metrics(metrics_list):
        return [np.mean([m[i] for m in metrics_list], axis=0) for i in range(5)]

    avg_d_one = avg_metrics(dmdc_one_list)
    avg_e_one = avg_metrics(edmdc_one_list)
    avg_d_multi = avg_metrics(dmdc_multi_list)
    avg_e_multi = avg_metrics(edmdc_multi_list)

    save_metric_tables_as_svg(_build_metric_dfs(avg_d_one, avg_e_one),
                              "Overall Average Metrics: One-Step Prediction",
                              save_dir / 'overall_metrics_one_step.svg')
    save_metric_tables_as_svg(_build_metric_dfs(avg_d_multi, avg_e_multi),
                              f"Overall Average Metrics: Multi-Step (p={step}) Prediction",
                              save_dir / 'overall_metrics_multi_step.svg')


# =============================================================================
# Plotting
# =============================================================================

def plot_predictions(results, save_dir: Path, step: int):
    """Generate 3x2 prediction plots for each trajectory."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 12,
        'axes.linewidth': 1.2
    })

    state_labels = ['u [m/s]', 'v [m/s]', 'w [m/s]',
                    'p [deg/s]', 'q [deg/s]', 'r [deg/s]']

    COLOR_TRUE = '#666666'
    COLOR_DMDC = '#005BBB'
    COLOR_EDMDC = '#D62728'

    for result in results:
        traj = result['trajectory']
        traj_name = result['traj_name']
        traj_dir = save_dir / traj_name
        traj_dir.mkdir(parents=True, exist_ok=True)

        # Resolve and normalize time axis
        if 'time' in traj.columns:
            time = traj['time'].values
        elif 'timestamp' in traj.columns:
            time = traj['timestamp'].values
        elif 't' in traj.columns:
            time = traj['t'].values
        else:
            time = np.arange(len(traj))
        time = time - time[0] if len(time) > 0 else time   # zero-base

        true_state = traj[[c for c in traj.columns
                          if c.startswith('odom_filtered.twist.twist')]].values

        def create_3x2_plot(pred_dmdc, pred_edmdc, title, filename):
            fig, axs = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
            fig.patch.set_facecolor('#ffffff')
            fig.suptitle(title, fontsize=16, weight='bold', color='#1a1a1a')

            for i in range(6):
                row, col = i % 3, i // 3
                ax = axs[row, col]

                ax.plot(time, true_state[:, i], label='Ground Truth',
                        color=COLOR_TRUE, linestyle='--', linewidth=1.5, zorder=1)
                ax.plot(time, pred_dmdc[:, i], label='DMDc',
                        color=COLOR_DMDC, linestyle='-', linewidth=1.5,
                        alpha=0.9, zorder=2)
                ax.plot(time, pred_edmdc[:, i], label='eDMDc',
                        color=COLOR_EDMDC, linestyle='-', linewidth=1.5,
                        alpha=0.9, zorder=3)

                ax.set_ylabel(state_labels[i])
                ax.grid(True, color='#e5e5e5', linestyle='-', linewidth=0.8, zorder=0)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(axis='both', which='both', direction='out', length=5)
                if row == 2:
                    ax.set_xlabel('Time [s]')

            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=3,
                       bbox_to_anchor=(0.5, 0.01), frameon=True,
                       facecolor='#ffffff', edgecolor='#dddddd')
            fig.align_ylabels()
            fig.tight_layout(rect=[0, 0.07, 1, 0.96], h_pad=1.5, w_pad=1.5)

            file_path = traj_dir / filename
            fig.savefig(file_path, format='svg', bbox_inches='tight', dpi=300)
            plt.close(fig)
            return file_path

        path1 = create_3x2_plot(
            result['dmdc_one_step_pred'], result['edmdc_one_step_pred'],
            'Koopman One-Step Prediction',
            'one_step_prediction.svg')
        sim_logger.info(f"Saved One-Step Plot to {path1}")

        path2 = create_3x2_plot(
            result['dmdc_multi_step_pred'], result['edmdc_multi_step_pred'],
            f'Koopman Multi-Step Prediction (p = {step})',
            'multi_step_prediction.svg')
        sim_logger.info(f"Saved Multi-Step Plot to {path2}")


# =============================================================================
# Main
# =============================================================================

def main():
    mlflow_uri = "https://mlflow.amarr.tan"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)

    dmdc_model = load_model(client=mlflow_client, name='dmdc', version=1)
    edmdc_model = load_model(client=mlflow_client, name='edmdc', version=1)

    dmdc_run = mlflow_client.get_run(
        mlflow_client.get_model_version(name='dmdc', version=1).run_id)
    edmdc_run = mlflow_client.get_run(
        mlflow_client.get_model_version(name='edmdc', version=1).run_id)

    dmdc_test_path, edmdc_test_path = [], []

    if len(dmdc_run.inputs.dataset_inputs) != len(edmdc_run.inputs.dataset_inputs):
        raise ValueError("Mismatched dataset inputs between DMDc and eDMDc models.")

    for dmdc_data, edmdc_data in zip(dmdc_run.inputs.dataset_inputs,
                                    edmdc_run.inputs.dataset_inputs):
        if dmdc_data.tags[0].value == 'test':
            dmdc_test_path.append(json.loads(dmdc_data.dataset.source)["uri"])
        if edmdc_data.tags[0].value == 'test':
            edmdc_test_path.append(json.loads(edmdc_data.dataset.source)["uri"])

    if set(dmdc_test_path) != set(edmdc_test_path):
        raise ValueError("Mismatched test dataset paths between DMDc and eDMDc models.")

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    http_client = urllib3.PoolManager(cert_reqs='CERT_NONE', assert_hostname=False)

    client = Minio(
        "s3.amarr.tan",
        access_key='minio_user',
        secret_key='minio_password',
        secure=True,
        http_client=http_client,
    )

    test_trajectories = []
    for path in dmdc_test_path:
        bucket_name, object_name = path.replace("s3://", "").split("/", 1)
        try:
            response = client.get_object(bucket_name, object_name)
            trajectory_data = pd.read_csv(response)
            response.close()
            response.release_conn()

            traj_name = object_name.replace('/', '_').replace('.csv', '')
            test_trajectories.append({'name': traj_name, 'data': trajectory_data})
        except S3Error as e:
            raise e

    results = []
    step = 10
    for traj_dict in test_trajectories:
        traj_name = traj_dict['name']
        traj = traj_dict['data'].copy()

        odom_cols = [c for c in traj.columns
                     if c.startswith('odom_filtered.twist.twist')]
        x = traj[odom_cols].values
        u = traj[[c for c in traj.columns if c.startswith('est_tau')]].values

        # Length sanity check
        if len(u) > len(x):
            sim_logger.warning(
                f"[{traj_name}] u has more rows than x ({len(u)} > {len(x)}); "
                f"truncating u.")
            u = u[:len(x)]

        dmdc_one_step_pred = one_step_prediction(dmdc_model, x, u)
        edmdc_one_step_pred = one_step_prediction(edmdc_model, x, u)
        dmdc_multi_step_pred = multi_step_prediction(dmdc_model, step, x, u)
        edmdc_multi_step_pred = multi_step_prediction(edmdc_model, step, x, u)

        # Convert angular velocity components from rad/s to deg/s for reporting
        if len(odom_cols) >= 6:
            traj.loc[:, odom_cols[3:6]] = np.rad2deg(traj[odom_cols[3:6]].values)
            for arr in (dmdc_one_step_pred, edmdc_one_step_pred,
                        dmdc_multi_step_pred, edmdc_multi_step_pred):
                arr[:, 3:6] = np.rad2deg(arr[:, 3:6])

        results.append({
            'traj_name': traj_name,
            'trajectory': traj,
            'dmdc_one_step_pred': dmdc_one_step_pred,
            'edmdc_one_step_pred': edmdc_one_step_pred,
            'dmdc_multi_step_pred': dmdc_multi_step_pred,
            'edmdc_multi_step_pred': edmdc_multi_step_pred,
        })

    base_result_dir = Path(
        "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/notebook/results/sysid")
    base_result_dir.mkdir(parents=True, exist_ok=True)

    report_metrics(results, save_dir=base_result_dir, step=step)
    plot_predictions(results, save_dir=base_result_dir, step=step)


if __name__ == "__main__":
    main()