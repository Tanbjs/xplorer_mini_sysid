import numpy as np

from ...utils.kinematic import cal_eta_err_with_ssa, eulerang

class PIDController:
    def __init__(self, kp, ki, kd, int_limit=None, sat_bound=None):
        self.kp = np.array(kp, dtype=float)
        self.ki = np.array(ki, dtype=float)
        self.kd = np.array(kd, dtype=float)
        
        self.integral = 0.0
        self.prev_error = None
        
        self.int_windup = np.array(int_limit, dtype=float) if int_limit is not None else None
        self.sat_bound = np.array(sat_bound, dtype=float) if sat_bound is not None else None

    @property
    def params(self):
        return {
            'kp': self.kp,
            'ki': self.ki,
            'kd': self.kd,
            'int_limit': self.int_windup,
            'sat_bound': self.sat_bound
        }

    def set_params(self, **kwargs):
        kp = kwargs.get('kp', None)
        ki = kwargs.get('ki', None)
        kd = kwargs.get('kd', None)
        int_limit = kwargs.get('int_limit', None)
        sat_bound = kwargs.get('sat_bound', None)

        if kp is not None:
            self.kp = np.array(kp, dtype=float)
        if ki is not None:
            self.ki = np.array(ki, dtype=float)
        if kd is not None:
            self.kd = np.array(kd, dtype=float)
        if int_limit is not None:
            self.int_windup = np.array(int_limit, dtype=float)
        if sat_bound is not None:
            self.sat_bound = np.array(sat_bound, dtype=float)

    def compute_control(self, measurement, setpoint, dt):
        if dt <= 0.0:
            return np.zeros_like(measurement)

        # 1. Compute Error
        error = np.array(setpoint) - np.array(measurement)
        
        # Initialize prev_error automatically in the first loop to match dimensions
        if self.prev_error is None:
            self.prev_error = np.zeros_like(error)

        # 2. Proportional
        p_term = self.kp * error
        
        # 3. Integral & Anti-windup
        self.integral += error * dt
        if self.int_windup is not None:
            self.integral = np.clip(self.integral, -self.int_windup, self.int_windup)
        i_term = self.ki * self.integral
        
        # 4. Derivative (Error Derivative)
        d_term = self.kd * ((error - self.prev_error) / dt)
        
        # 5. Output sum
        u = p_term + i_term + d_term
        
        # 6. Output Saturation
        if self.sat_bound is not None:
            u = np.clip(u, -self.sat_bound, self.sat_bound)
            
        # Update State
        self.prev_error = error
        
        return u
        
    def reset(self):
        self.integral = 0.0
        self.prev_error = None


class PositionPIDController(PIDController):
    def __init__(self, kp, ki, kd, int_limit=None, sat_bound=None):
        super().__init__(kp, ki, kd, int_limit, sat_bound)
        self.integral = np.zeros(6)
        self.prev_error = np.zeros(6)

    def compute_control(self, eta, eta_ref, dt):
        if dt <= 0.0: 
            return np.zeros(6)

        eta_error = cal_eta_err_with_ssa(eta_ref, eta)
        
        derivative = (eta_error - self.prev_error) / dt
        J, _, _ = eulerang(eta[3], eta[4], eta[5])
        nu_cmd_unsat = np.linalg.inv(J) @ ((self.kp * eta_error) + (self.ki * self.integral) + (self.kd * derivative))

        # Saturation & Conditional Anti-windup
        if self.sat_bound is not None:
            nu_cmd_b = np.clip(nu_cmd_unsat, -self.sat_bound, self.sat_bound)
            
            is_saturated = (nu_cmd_unsat > self.sat_bound) | (nu_cmd_unsat < -self.sat_bound)
            same_direction = (np.sign(eta_error) == np.sign(nu_cmd_unsat))
            stop_integrating = is_saturated & same_direction
            self.integral += (~stop_integrating) * (eta_error * dt)
        else:
            nu_cmd_b = nu_cmd_unsat
            self.integral += eta_error * dt
        
        # Global Integral limit
        if self.int_windup is not None:
            self.integral = np.clip(self.integral, -self.int_windup, self.int_windup)

        self.prev_error = eta_error

        return nu_cmd_b


class VelocityPIDController(PIDController):
    def __init__(self, kp, ki, kd, int_limit=None, sat_bound=None):
        super().__init__(kp, ki, kd, int_limit, sat_bound)
        self.mrb = np.array([
            [55.0,  0.0,  0.0,  0.0,      0.0,      0.0],     # Surge
            [ 0.0, 55.0,  0.0,  0.0,      0.0,      0.0],     # Sway
            [ 0.0,  0.0, 55.0,  0.0,      0.0,      0.0],     # Heave
            [ 0.0,  0.0,  0.0,  4.8,     -0.00598,  0.01147], # Roll
            [ 0.0,  0.0,  0.0, -0.00598,  6.3,      0.00081], # Pitch
            [ 0.0,  0.0,  0.0,  0.01147,  0.00081,  2.1]      # Yaw
        ])
        self.integral = np.zeros(6)
        self.prev_error = np.zeros(6)

    def compute_control(self, nu, nu_ref, dt):
        if dt <= 0.0: return np.zeros(6)

        nu_error = np.array(nu_ref) - np.array(nu)
        derivative = (nu_error - self.prev_error) / dt
        tau_cmd_unsat = self.mrb @ ((self.kp * nu_error) + (self.ki * self.integral) + (self.kd * derivative))

        if self.sat_bound is not None:
            tau_cmd_b = np.clip(tau_cmd_unsat, -self.sat_bound, self.sat_bound)
            # Conditional Anti-windup (Refactored)
            is_saturated = (tau_cmd_unsat > self.sat_bound) | (tau_cmd_unsat < -self.sat_bound)
            same_direction = (np.sign(nu_error) == np.sign(tau_cmd_unsat))
            stop_integrating = is_saturated & same_direction
            self.integral += (~stop_integrating) * (nu_error * dt)
        else:
            tau_cmd_b = tau_cmd_unsat
            self.integral += nu_error * dt

        if self.int_windup is not None:
            self.integral = np.clip(self.integral, -self.int_windup, self.int_windup)

        self.prev_error = nu_error
        return tau_cmd_b