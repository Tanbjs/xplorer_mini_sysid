from typing import Union, TypeAlias
import inspect
import numpy as np

from kmc.utils.model_wrapper import DMDcWrapper, EDMDcWrapper, DeepModelWrapper

from ..core.params import Weights, Bounds
from ..core.model import KoopmanModel
from ...lib.controller import pid, lqr, mpc


Wrapper: TypeAlias = Union[DMDcWrapper, EDMDcWrapper, DeepModelWrapper]
PositionControllerType: TypeAlias = Union[pid.PositionPIDController, pid.PositionFFPIController]
VelocityControllerType: TypeAlias = Union[pid.VelocityPIDController, 
                                          lqr.StandardStateForm, lqr.StandardOutputForm, 
                                          mpc.StandardStateForm, mpc.StandardOutputForm,
                                          mpc.IncrementalStateForm, mpc.IncrementalOutputForm]


def create_position_controller(**kwargs) -> PositionControllerType:

    params = {k.replace('params.', ''): v.value for k, v in kwargs.items() if k.startswith('params.')}
    ctrl_type = kwargs.get('type').value

    if ctrl_type == 'pid':
        return pid.PositionPIDController(**params)
    
    elif ctrl_type == 'ff_pi':
        return pid.PositionFFPIController(**params)

    else:
        raise ValueError(f"Unsupported controller type: {ctrl_type}")
    
def create_velocity_controller(model: Wrapper = None, 
                               node_name: str = None,
                               dt: float = None,
                               logger=None, 
                               **kwargs) -> VelocityControllerType:

    def get_val(key, default=None):
        p = kwargs.get(key)
        return p.value if p is not None else default

    ctrl_type = get_val('type')
    mode = get_val('mode')
    use_preview = get_val('use_preview', False)

    param = {k.replace('params.', ''): v.value for k, v in kwargs.items() if k.startswith('params.')}

    if ctrl_type == 'pid':
        return pid.VelocityPIDController(**param)
    
    def extract_from_signature(cls, source_dict):
        sig_keys = inspect.signature(cls).parameters.keys()
        return {k: np.array(source_dict.pop(k)) for k in sig_keys if k in source_dict}

    if ctrl_type == 'lqr_standard':
        weights_data = extract_from_signature(Weights, param)
        q_vec = weights_data.get('Q')
        r_vec = weights_data.get('R_abs')
        
        if q_vec is None or r_vec is None:
            raise ValueError(f"LQR requires Q and R weights in params.")

        weights = Weights(Q=np.diag(q_vec), R_abs=np.diag(r_vec))
        lqr_params = lqr.LQRParams(weights=weights)

        if mode == 'state_form':
            return lqr.StandardStateForm(model, lqr_params, use_preview=use_preview, logger=logger)
        elif mode == 'output_form':
            return lqr.StandardOutputForm(model, lqr_params, use_preview=use_preview, logger=logger)
    
    elif ctrl_type == 'mpc_standard':
        n_horizon = int(param.pop('N_horizon', 10))
        weights_dict = extract_from_signature(Weights, param)
        bounds_dict = extract_from_signature(Bounds, param)

        weights = Weights(Q=np.diag(weights_dict.get('Q', None)) if weights_dict.get('Q', None) is not None else None, 
                          R_abs=np.diag(weights_dict.get('R_abs', None)) if weights_dict.get('R_abs', None) is not None else None,
                          R_rate= np.diag(weights_dict.get('R_rate', None)) if weights_dict.get('R_rate', None) is not None else None
                        )
                          
        bounds = Bounds(x_min=np.array(bounds_dict.get('x_min', None)),
                        x_max=np.array(bounds_dict.get('x_max', None)),
                        y_min=np.array(bounds_dict.get('y_min', None)),
                        y_max=np.array(bounds_dict.get('y_max', None)),
                        u_min=np.array(bounds_dict.get('u_min', None)),
                        u_max=np.array(bounds_dict.get('u_max', None)),
                        du_min=np.array(bounds_dict.get('du_min', None)),
                        du_max=np.array(bounds_dict.get('du_max', None))
                    )

        mpc_params = mpc.MPCParams(dt=dt, N_horizon=n_horizon, weights=weights, bounds=bounds)

        if mode == 'state_form':
            return mpc.StandardStateForm(model, mpc_params, node_name=node_name, 
                                         use_preview=use_preview, logger=logger)
        
        elif mode == 'output_form':
            return mpc.StandardOutputForm(model, mpc_params, node_name=node_name, 
                                          use_preview=use_preview, logger=logger)
        else:
            raise ValueError(f"MPC requires mode 'state_form' or 'output_form', got: {mode}")

    raise ValueError(f"Unsupported controller type: {ctrl_type}")