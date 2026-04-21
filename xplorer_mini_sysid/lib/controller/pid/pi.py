import numpy as np
from ...utils.kinematic import cal_eta_err_with_ssa, eulerang


class PositionFFPIController:
    def __init__(self, kp, ki, int_limit=None, sat_bound=None):
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.int_limit = np.array(int_limit) if int_limit is not None else None
        self.sat_bound = np.array(sat_bound) if sat_bound is not None else None
        self.integral = np.zeros(6)
    
    @property
    def params(self):
        return {
            'kp': self.kp,
            'ki': self.ki,
            'int_limit': self.int_limit,
            'sat_bound': self.sat_bound
        }
    
    def set_params(self, **kwargs):
        if 'kp' in kwargs:
            self.kp = np.array(kwargs['kp'])
        if 'ki' in kwargs:
            self.ki = np.array(kwargs['ki'])
        if 'int_limit' in kwargs:
            val = kwargs['int_limit']
            self.int_limit = np.array(val) if val is not None else None
        if 'sat_bound' in kwargs:
            val = kwargs['sat_bound']
            self.sat_bound = np.array(val) if val is not None else None

    def compute_control(self, eta, eta_ref, dt, eta_ref_dot=None):
            if dt <= 0.0:
                return np.zeros(6)

            # 1. Position Error in NED
            eta_error_n = cal_eta_err_with_ssa(eta_ref, eta)
            
            # 2. Feed-forward in NED
            ff_n = np.array(eta_ref_dot) if eta_ref_dot is not None else np.zeros(6)

            # 3. Control Law in NED Frame (P + I + FF)
            nu_cmd_n_unsat = ff_n + (self.kp * eta_error_n) + (self.ki * self.integral)

            # 4. Coordinate Transformation (NED -> Body)
            J, _, _ = eulerang(eta[3], eta[4], eta[5])
            inv_J = np.linalg.pinv(J)
            
            # Multiply J^-1 to the entire sum
            nu_cmd_b_unsat = inv_J @ nu_cmd_n_unsat

            if self.sat_bound is not None:
                nu_cmd_b_sat = np.clip(nu_cmd_b_unsat, -self.sat_bound, self.sat_bound)
                
                # --- Back-Calculation Anti-Windup ---
                delta_nu_b = nu_cmd_b_unsat - nu_cmd_b_sat
                delta_nu_n = J @ delta_nu_b
                self.integral += (eta_error_n * dt) - (delta_nu_n * dt)
                
                nu_cmd_b = nu_cmd_b_sat
            else:
                nu_cmd_b = nu_cmd_b_unsat
                self.integral += eta_error_n * dt

            # 5. Global Integral Limit (Safety Bounding)
            if self.int_limit is not None:
                self.integral = np.clip(self.integral, -self.int_limit, self.int_limit)

            return nu_cmd_b