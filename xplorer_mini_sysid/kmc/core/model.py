import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass 
class LinearModel:
    """
    Linear Model representation for MPC.
        A: State transition matrix
        B: Control input matrix
        C: Output matrix
        D: Feedthrough matrix (optional)
    """
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: Optional[np.ndarray] = None


@dataclass
class KoopmanModel:
    """
    Koopman Model representation for MPC.
        dyn: LinearModel - The linear dynamics model of the system.
        lift: Callable - The lifting function to map the original state to the lifted space.
        scaler_x: Optional[object] - Scaler for the state variables (optional).
        scaler_u: Optional[object] - Scaler for the control inputs (optional).
        scaler_y: Optional[object] - Scaler for the output variables (optional).
    """
    dyn: LinearModel
    lift: Callable
    scaler_x: Optional[object] = None
    scaler_u: Optional[object] = None
    scaler_y: Optional[object] = None