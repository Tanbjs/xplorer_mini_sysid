from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Weights:
    """
    Weights for the MPC cost function.
        Q: State tracking cost weight matrix
        R_abs: Control effort cost weight matrix
        R_rate: Control rate cost weight matrix (optional)
        P: Terminal cost weight matrix (optional)
        S: Cross-term cost weight matrix (optional)
    """
    Q: np.ndarray  
    R_abs: Optional[np.ndarray] = None
    R_rate: Optional[np.ndarray] = None 
    P: Optional[np.ndarray] = None
    S: Optional[np.ndarray] = None


@dataclass
class Bounds:
    """
    Bounds for the MPC optimization variables.
        x_min: Minimum state values (optional)
        x_max: Maximum state values (optional)
        y_min: Minimum output values (optional)
        y_max: Maximum output values (optional)
        u_min: Minimum control input values (optional)
        u_max: Maximum control input values (optional)
        du_min: Minimum control input rate values (optional)
        du_max: Maximum control input rate values (optional)
        terminal_bound_min: Minimum terminal state values (optional)
        terminal_bound_max: Maximum terminal state values (optional)
    """
    x_min: Optional[np.ndarray] = None
    x_max: Optional[np.ndarray] = None
    y_min: Optional[np.ndarray] = None
    y_max: Optional[np.ndarray] = None
    u_min: Optional[np.ndarray] = None
    u_max: Optional[np.ndarray] = None
    du_min: Optional[np.ndarray] = None
    du_max: Optional[np.ndarray] = None
    terminal_bound_min: Optional[np.ndarray] = None
    terminal_bound_max: Optional[np.ndarray] = None