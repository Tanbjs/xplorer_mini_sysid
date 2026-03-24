from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from kmc.utils.model_wrapper import DMDcWrapper, EDMDcWrapper, DeepModelWrapper

from ...core.kmc import BaseKMC
from ...core.model import KoopmanModel
from ...core.params import Weights


@dataclass
class LQRParams:
    """
    Parameters for LQR setup.
        dt: Discretization time step
        N_horizon: Prediction horizon length
    """
    weights: Weights


class KLQR(BaseKMC):
    
    """
    Base class for Koopman-based LQR controllers.
    Subclasses must implement the compute_control method to calculate the control input based on the current state and reference.
    
    Attributes:
        lqr_params: LQRParams - Parameters for LQR setup, including cost weights and horizon length.
        model: KoopmanModel - The Koopman model of the system, containing the system dynamics and lifting function.

    Methods:
        set_params(**kwargs): Update LQR parameters dynamically. Accepts any parameter defined in LQRParams or its nested Weights dataclass.
        compute_control(x, y_ref): Abstract method to compute the control input based on the current state x and reference output y_ref. Must be implemented by subclasses.
    """

    lqr_params: LQRParams
    model: KoopmanModel

    def __init__(self, 
                 model_wrapper: DMDcWrapper | EDMDcWrapper | DeepModelWrapper,
                 lqr_params: LQRParams):
        
        super().__init__(model_wrapper)
        self.lqr_params = lqr_params

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.lqr_params, key):
                setattr(self.lqr_params, key, value)
            elif hasattr(self.lqr_params.weights, key):
                setattr(self.lqr_params.weights, key, value)
            else:
                raise ValueError(f"Invalid parameter name: {key}")

    @abstractmethod
    def compute_control(self, x: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement compute_control method.")


    
