import numpy as np
from ...utils.kinematic import cal_eta_err_with_ssa, eulerang


class PositionFFPIController:
    def __init__(self, kp, ki, int_limit=None, sat_bound=None):
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.int_limit = np.array(int_limit) if int_limit is not None else np.ones(6)
        self.sat_bound = np.array(sat_bound) if sat_bound is not None else np.ones(6)
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
        kp = kwargs.get('kp')
        ki = kwargs.get('ki')
        int_limit = kwargs.get('int_limit')
        sat_bound = kwargs.get('sat_bound')

        if kp is not None:
            self.kp = np.array(kp)
        if ki is not None:
            self.ki = np.array(ki)
        if int_limit is not None:
            self.int_limit = np.array(int_limit)
        if sat_bound is not None:
            self.sat_bound = np.array(sat_bound)

    def compute_control(self, eta, eta_ref, dt, eta_ref_dot=None):
        # 1. Position Error in NED
        eta_error_n = cal_eta_err_with_ssa(eta_ref, eta)

        # 2. Coordinate Transformation (NED -> Body)
        J, _, _ = eulerang(eta[3], eta[4], eta[5])
        inv_J = np.linalg.pinv(J)
        
        # Error and Feed-forward in Body Frame
        error_b = inv_J @ eta_error_n
        
        if eta_ref_dot is None:
            ff_b = np.zeros(6)
        else:
            ff_b = inv_J @ np.array(eta_ref_dot)

        # 3. Control Law (Body Frame)
        # u_b = Kp*e_b + Ki*integral_b + ff_b
        nu_cmd_unsat = (self.kp * error_b) + (self.ki * self.integral) + ff_b
        nu_cmd_b = np.clip(nu_cmd_unsat, -self.sat_bound, self.sat_bound)

        # 4. Anti-windup: Clamping logic
        is_saturated = np.abs(nu_cmd_unsat) > self.sat_bound
        same_direction = np.sign(error_b) == np.sign(nu_cmd_unsat)
        stop_integrating = is_saturated & same_direction

        # Update Integral state in Body Frame
        self.integral += (~stop_integrating) * (error_b * dt)
        self.integral = np.clip(self.integral, -self.int_limit, self.int_limit)

        return nu_cmd_b