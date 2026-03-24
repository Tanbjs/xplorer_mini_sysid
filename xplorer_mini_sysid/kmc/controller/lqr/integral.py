from .base import KLQR, LQRParams
from kmc.utils.model_wrapper import DMDcWrapper, EDMDcWrapper, DeepModelWrapper

class Integral(KLQR):
    def __init__(self, 
                 model: DMDcWrapper | EDMDcWrapper | DeepModelWrapper, 
                 params: LQRParams):
        
        super().__init__(params)
        self.integral_error = None

    def compute_control(self, x, y_ref):
        pass