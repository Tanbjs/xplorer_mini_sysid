from typing import Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from acados_template import AcadosOcpSolver, AcadosModel
from kmc.utils.model_wrapper import DMDcWrapper, EDMDcWrapper, DeepModelWrapper


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
    """
    x_min: Optional[np.ndarray] = None
    x_max: Optional[np.ndarray] = None
    y_min: Optional[np.ndarray] = None
    y_max: Optional[np.ndarray] = None
    u_min: Optional[np.ndarray] = None
    u_max: Optional[np.ndarray] = None
    du_min: Optional[np.ndarray] = None
    du_max: Optional[np.ndarray] = None


@dataclass
class MPCParams:
    """
    Parameters for MPC setup.
        dt: Discretization time step
        N_horizon: Prediction horizon length
    """
    dt: float
    N_horizon: int
    weights: Weights
    bounds: Bounds


@dataclass 
class LinearModel:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: Optional[np.ndarray] = None


@dataclass
class KoopmanModel:
    """
    Linear Model representation for MPC.
        A: State transition matrix
        B: Control input matrix
        C: Output matrix
        D: Feedthrough matrix (optional)
    """
    dyn: LinearModel
    lift: Callable
    scaler_x: Optional[object] = None
    scaler_u: Optional[object] = None
    scaler_y: Optional[object] = None


class KMC(ABC):

    mpc_params: MPCParams
    model: KoopmanModel

    def __init__(self, 
                 model_wrapper: DMDcWrapper | EDMDcWrapper | DeepModelWrapper,
                 mpc_params: MPCParams):
        self.model = self.__get_model(model_wrapper)
        self.mpc_params = mpc_params 

    def __get_model(self, model_wrapper: DMDcWrapper | EDMDcWrapper | DeepModelWrapper):

        if isinstance(model_wrapper, DMDcWrapper):
            lift_func = lambda x: x 
        elif isinstance(model_wrapper, EDMDcWrapper):
            lift_func = lambda x: model_wrapper.model._obs_func.transform(x).reshape(1,-1)
        elif isinstance(model_wrapper, DeepModelWrapper):
            lift_func = lambda x: model_wrapper.model.model.lift(x).detach().cpu().numpy()
        else:
            raise ValueError("Unsupported model wrapper type.")

        return KoopmanModel(
                        dyn=LinearModel(
                            A=model_wrapper.A,
                            B=model_wrapper.B,
                            C=model_wrapper.C,
                            D=getattr(model_wrapper, 'D', None)
                        ),
                        lift=lift_func,
                        scaler_x=getattr(model_wrapper, 'scaler_x', None),
                        scaler_u=getattr(model_wrapper, 'scaler_u', None),
                        scaler_y=getattr(model_wrapper, 'scaler_y', None))

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.mpc_params, key):
                setattr(self.mpc_params, key, value)
            elif hasattr(self.mpc_params.weights, key):
                setattr(self.mpc_params.weights, key, value)
            elif hasattr(self.mpc_params.bounds, key):
                setattr(self.mpc_params.bounds, key, value)
            else:
                raise ValueError(f"Invalid parameter name: {key}")

    @abstractmethod
    def compute_control(self) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement compute_control method.")


    
