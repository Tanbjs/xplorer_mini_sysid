from dataclasses import dataclass
from typing import Optional

import numpy as np
from tabulate import tabulate


@dataclass
class Weights:
    Q: np.ndarray  
    Qi: Optional[np.ndarray] = None
    R_abs: Optional[np.ndarray] = None
    R_rate: Optional[np.ndarray] = None 
    P: Optional[np.ndarray] = None
    S: Optional[np.ndarray] = None

    def __post_init__(self):
        self.log_summary()

    def log_summary(self):
        print(f"\n{'='*25} MPC WEIGHTS SETTINGS {'='*25}")
        table_data = [
            ["Q (State)", "Active" if self.Q is not None else "None", 
             f"{self.Q.shape[0]}x{self.Q.shape[1]}" if self.Q is not None else "", 
             np.diag(self.Q).round(4).tolist() if self.Q is not None else ""],
            ["Qi (Integ)", "Active" if self.Qi is not None else "None", 
             f"{self.Qi.shape[0]}x{self.Qi.shape[1]}" if self.Qi is not None else "", 
             np.diag(self.Qi).round(4).tolist() if self.Qi is not None else ""],
            ["R_abs (Effort)", "Active" if self.R_abs is not None else "None", 
             f"{self.R_abs.shape[0]}x{self.R_abs.shape[1]}" if self.R_abs is not None else "", 
             np.diag(self.R_abs).round(4).tolist() if self.R_abs is not None else ""],
            ["R_rate (Rate)", "Active" if self.R_rate is not None else "None", 
             f"{self.R_rate.shape[0]}x{self.R_rate.shape[1]}" if self.R_rate is not None else "", 
             np.diag(self.R_rate).round(4).tolist() if self.R_rate is not None else ""],
            ["P (Terminal)", "Active" if self.P is not None else "None", 
             f"{self.P.shape[0]}x{self.P.shape[1]}" if self.P is not None else "", 
             np.diag(self.P).round(4).tolist() if self.P is not None else ""],
            ["S (Cross)", "Active" if self.S is not None else "None", 
             f"{self.S.shape[0]}x{self.S.shape[1]}" if self.S is not None else "", 
             "Non-square" if self.S is not None else ""]
        ]
        headers = ["Matrix", "Status", "Dim", "Diagonal Values"]
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))


@dataclass
class Bounds:
    x_min: Optional[np.ndarray] = None; x_max: Optional[np.ndarray] = None
    y_min: Optional[np.ndarray] = None; y_max: Optional[np.ndarray] = None
    u_min: Optional[np.ndarray] = None; u_max: Optional[np.ndarray] = None
    du_min: Optional[np.ndarray] = None; du_max: Optional[np.ndarray] = None
    terminal_bound_min: Optional[np.ndarray] = None; terminal_bound_max: Optional[np.ndarray] = None

    def __post_init__(self):
        self.log_summary()

    def log_summary(self):
        print(f"\n{'='*25} MPC BOUNDS SETTINGS {'='*25}")
        groups = [
            ("State (x)", self.x_min, self.x_max),
            ("Output (y)", self.y_min, self.y_max),
            ("Input (u)", self.u_min, self.u_max),
            ("Rate (du)", self.du_min, self.du_max),
            ("Terminal", self.terminal_bound_min, self.terminal_bound_max),
        ]
        table_data = []
        for name, v_min, v_max in groups:
            if v_min is not None or v_max is not None:
                dim = f"{v_min.shape[0]}" if v_min is not None else f"{v_max.shape[0]}"
                v_min_str = v_min.round(3).tolist() if v_min is not None else "-inf"
                v_max_str = v_max.round(3).tolist() if v_max is not None else "+inf"
                table_data.append([name, "Active", dim, v_min_str, v_max_str])
            else:
                table_data.append([name, "None", "", "", ""])

        headers = ["Constraint", "Status", "Dim", "Min Bound", "Max Bound"]
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))