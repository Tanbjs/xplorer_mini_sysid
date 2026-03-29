from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from kmc.utils.model_wrapper import DMDcWrapper, EDMDcWrapper, DeepModelWrapper

from ...core.kmc import BaseKMC
from ...core.model import KoopmanModel
from ...core.params import Weights, Bounds


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


class KMPC(BaseKMC):
    
    """
    Base class for Koopman-based MPC controllers.
    Subclasses must implement the compute_control method to calculate the control input based on the current state and reference.
    
    Attributes:
        mpc_params: MPCParams - Parameters for MPC setup, including cost weights and horizon length.
        model: KoopmanModel - The Koopman model of the system, containing the system dynamics and lifting function.

    Methods:
        set_params(**kwargs): Update MPC parameters dynamically. Accepts any parameter defined in MPCParams or its nested Weights dataclass.
        compute_control(x, y_ref): Abstract method to compute the control input based on the current state x and reference output y_ref. Must be implemented by subclasses.
    """

    mpc_params: MPCParams
    model: KoopmanModel

    def __init__(self, 
                 model_wrapper: DMDcWrapper | EDMDcWrapper | DeepModelWrapper,
                 mpc_params: MPCParams):
        
        super().__init__(model_wrapper)
        self.mpc_params = mpc_params

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
    def compute_control(self, x: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement compute_control method.")


    
