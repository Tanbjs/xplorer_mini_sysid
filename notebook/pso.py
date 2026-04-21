import os
import copy
import warnings
import urllib3
import logging
import numpy as np
import yaml

import mlflow
import pyswarms as ps
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import plot_cost_history

from xplorer_mini_sysid.lib.utils.mlflow import load_model 

warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
sim_logger = logging.getLogger("Simulation")
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

from simulation import (run_cascade_simulation, 
                        run_velocity_simulation,
                        load_model, load_params,
                        get_config_flags, 
                        init_controllers, 
                        figure8_path)

def generate_nu_ref(t_end, dt, test_profiles):
    """
    สร้าง nu_ref_full array (N x 6) สำหรับการทำ Velocity Inner-loop Tuning
    
    Args:
        t_end (float): เวลาสิ้นสุด simulation (วินาที)
        dt (float): Time step
        test_profiles (list of dict): กำหนดรูปแบบ signal ในแต่ละ DOF
    """
    num_steps = int(t_end / dt)
    t = np.linspace(0, t_end, num_steps, endpoint=False)
    nu_ref_full = np.zeros((num_steps, 6))
    
    for profile in test_profiles:
        dof = profile.get('dof', 0)     # 0:u (Surge), 1:v (Sway), 2:w (Heave), 3:p (Roll), 4:q (Pitch), 5:r (Yaw)
        sig_type = profile.get('type', 'step')
        mag = profile.get('mag', 1.0)
        t_start = profile.get('t_start', 0.0)
        
        idx_start = int(t_start / dt)
        
        if sig_type == 'step':
            nu_ref_full[idx_start:, dof] += mag
            
        elif sig_type == 'sine':
            freq = profile.get('freq', 0.1) 
            nu_ref_full[idx_start:, dof] += mag * np.sin(2 * np.pi * freq * (t[idx_start:] - t_start))
            
        elif sig_type == 'ramp':
            t_ramp = profile.get('t_ramp', 5.0) 
            idx_ramp_end = min(int((t_start + t_ramp) / dt), num_steps)
            
            slope = mag / t_ramp
            nu_ref_full[idx_start:idx_ramp_end, dof] += slope * (t[idx_start:idx_ramp_end] - t_start)

            if idx_ramp_end < num_steps:
                nu_ref_full[idx_ramp_end:, dof] += mag

    return nu_ref_full

def main():
    mlflow_uri = "https://mlflow.amarr.tan" 
    mlflow.set_tracking_uri(mlflow_uri)

    ctrl_path = "/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_sysid/config/controller/dmdc/without_preview/ffpi_stdmpc_gain.yaml"
    auv_params = load_params('/home/tanbjs/xplorer_mini_sim_ws/src/xplorer_mini_descriptions/robots/xplorer_mini_dynamic_parameters.yaml')

    with open(ctrl_path, 'r') as f:
        ros_params = yaml.safe_load(f)['/**']['ros__parameters']

    mlflow_client = mlflow.MlflowClient()
    koopman_model = load_model(client=mlflow_client, name=ros_params['model_name'], version=ros_params['model_version'])

    # print initial Q and R_abs
    initial_Q = ros_params['velocity_controller']['params']['Q']
    initial_R_abs = ros_params['velocity_controller']['params']['R_abs']
    print("\n" + "="*50)
    print(" INITIAL CONTROLLER GAINS")
    print("="*50)
    print(f"Initial Q:      {[round(num, 2) for num in initial_Q]}")
    print(f"Initial R_abs:  {[round(num, 2) for num in initial_R_abs]}") 


    # 1. Generate Trajectory
    dt = 0.1
    t_end = 60
    nu_ref = generate_nu_ref(
        t_end=t_end,
        dt=dt,
        test_profiles=[
            {'dof': 0, 'type': 'step', 'mag': 0.5, 't_start': 5.0},   # Surge step at 5s
            {'dof': 1, 'type': 'sine', 'mag': 0.3, 'freq': 0.05, 't_start': 10.0}, # Sway sine wave starting at 10s
            {'dof': 2, 'type': 'ramp', 'mag': 0.4, 't_start': 15.0, 't_ramp': 10.0}, # Heave ramp starting at 15s
            {'dof': 5, 'type': 'sine', 'mag': 0.2, 'freq': 0.1, 't_start': 20.0} # Yaw sine wave starting at 20s
        ]
    )
    
    flags = get_config_flags(ros_params)

    # ==========================================
    # 2. Define PSO Objective Function (Closure)
    # ==========================================
    
    def evaluate_swarm(particles):
        """
        รับค่าอนุภาคทั้งหมด (n_particles, 12) รีเทิร์นค่า cost (n_particles,)
        """
        n_particles = particles.shape[0]
        costs = np.zeros(n_particles)
        
        for i in range(n_particles):
            # แยกค่า Q (6 ตัวแรก) และ R_abs (6 ตัวหลัง)
            Q_cand = particles[i, 0:6].tolist()
            R_abs_cand = particles[i, 6:12].tolist()
            
            # Deepcopy เพื่อไม่ให้ params ชนกันระหว่างลูป
            temp_params = copy.deepcopy(ros_params)
            temp_params['velocity_controller']['params']['Q'] = Q_cand
            temp_params['velocity_controller']['params']['R_abs'] = R_abs_cand
            
            # Re-initialize controller ด้วยค่า Gain ใหม่ (ไม่ต้องใช้ pose_ctrl)
            _, vel_ctrl = init_controllers(koopman_model, temp_params, sim_logger)
            
            # Run Isolated Velocity Simulation 
            history = run_velocity_simulation(auv_params, vel_ctrl, nu_ref, flags, t_end, dt, DEBUG_MODE=False)
            
            # -----------------------------------------------------
            # [FIXED] คำนวณ Cost (Fitness) ด้วย Velocity Tracking Error (nu)
            # -----------------------------------------------------
            # state x คือ [eta(0:6), nu(6:12)] -> ดึงแค่ nu มาประเมิน
            nu_actual = np.array(history['x'])[:, 6:12]
            
            # ป้องกันกรณี history สั้นกว่า nu_ref
            length = min(len(nu_actual), len(nu_ref))
            e_nu = nu_ref[:length, :] - nu_actual[:length, :]
            
            # Calculate RMSE ของ ความเร็ว
            rmse = np.sqrt(np.mean(e_nu**2))
            
            # [Optional] Penalize Actuator Effort (ป้องกัน Controller สั่งงานกระชาก)
            # tau = np.array(history['tau'])[:length, :]
            # rms_tau = np.sqrt(np.mean(tau**2))
            # costs[i] = rmse + (0.001 * rms_tau) # Weight penalty ตามความเหมาะสม
            
            costs[i] = rmse
            
        return costs

    # ==========================================
    # 3. Setup PSO & Run
    # ==========================================
    sim_logger.info("========== Starting PSO Optimization ==========")
    
    # กำหนดขอบเขตการสุ่ม (Bounds)
    # Q_min, Q_max = 1.0, 500.0 | R_abs_min, R_abs_max = 0.001, 1.0
    min_bound = np.array([0.1]*6 + [0.01]*6)
    max_bound = np.array([5000.0]*6 + [1.0]*6)
    bounds = (min_bound, max_bound)
    
    # พารามิเตอร์กลไก PSO (c1: Cognitive, c2: Social, w: Inertia)
    options = {'c1': 2.0, 'c2': 1.0, 'w': 0.85, 'k': 3, 'p': 2}
    
    # สร้าง Optimizer (12 มิติ = Q 6 ตัว + R 6 ตัว)
    # หมายเหตุ: n_particles และ iters ส่งผลโดยตรงต่อระยะเวลาการรัน
    # optimizer = ps.single.LocalBestPSO(n_particles=40, dimensions=12, options=options, bounds=bounds)
    optimizer = ps.single.GlobalBestPSO(n_particles=40, dimensions=12, options=options, bounds=bounds)
    
    # สั่งรัน Optimization
    best_cost, best_pos = optimizer.optimize(evaluate_swarm, iters=20)
    
    # ==========================================
    # 4. Extract and Report Best Parameters
    # ==========================================
    best_Q = best_pos[0:6].tolist()
    best_R_abs = best_pos[6:12].tolist()
    
    print("\n" + "="*50)
    print(" PSO OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Best Cost (RMSE): {best_cost:.4f}")
    print(f"Optimized Q:      {[round(num, 2) for num in best_Q]}")
    print(f"Optimized R_abs:  {[round(num, 2) for num in best_R_abs]}")
    
    plot_cost_history(cost_history=optimizer.cost_history)
    plt.title("PSO Convergence History")
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost (RMSE)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()