from typing import Union, TypeAlias
import inspect
import json
import numpy as np

from kmc.utils.model_wrapper import DMDcWrapper, EDMDcWrapper, DeepModelWrapper

from ..core.params import Weights, Bounds
from ...lib.controller import pid, mpc


Wrapper: TypeAlias = Union[DMDcWrapper, EDMDcWrapper, DeepModelWrapper]
PositionControllerType: TypeAlias = Union[pid.PositionPIDController, pid.PositionFFPIController]
VelocityControllerType: TypeAlias = Union[pid.VelocityPIDController, 
                                          mpc.UnconstrainedStateForm, mpc.UnconstrainedOutputForm, 
                                          mpc.UnconstrainedIntegralStateForm,
                                          mpc.ConstrainedStandardStateForm, mpc.ConstrainedStandardOutputForm,
                                          mpc.ConstrainedIntegralStateForm,
                                          mpc.IncrementalStateForm, mpc.IncrementalOutputForm]


def create_position_controller(logger=None, **kwargs) -> PositionControllerType:

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

    def extract_from_signature(cls, source_dict):
        sig_keys = inspect.signature(cls).parameters.keys()
        return {k: np.array(source_dict.pop(k)) for k in sig_keys if k in source_dict}
    
    def safe_array(val):
        return np.array(val) if val is not None else None
    
    # get params
    ctrl_type = get_val('type')
    use_constraints = get_val('use_constraints', False)
    mode = get_val('mode')
    use_preview = get_val('use_preview', False)
    params = {k.replace('params.', ''): v.value for k, v in kwargs.items() if k.startswith('params.')}
    
    if ctrl_type == 'pid':
        return pid.VelocityPIDController(**params)

    elif 'mpc' in ctrl_type:
        
        n_horizon = int(params.pop('N_horizon', 10))
        weights_dict = extract_from_signature(Weights, params)
        bounds_dict = extract_from_signature(Bounds, params)

        weights = Weights(
            Q=np.diag(weights_dict.get('Q')) if weights_dict.get('Q') is not None else None, 
            Qi=np.diag(weights_dict.get('Qi')) if weights_dict.get('Qi') is not None else None,
            R_abs=np.diag(weights_dict.get('R_abs')) if weights_dict.get('R_abs') is not None else None,
            R_rate=np.diag(weights_dict.get('R_rate')) if weights_dict.get('R_rate') is not None else None
        )
                        
        bounds = Bounds(x_min=safe_array(bounds_dict.get('x_min', None)),
                        x_max=safe_array(bounds_dict.get('x_max', None)),
                        y_min=safe_array(bounds_dict.get('y_min', None)),
                        y_max=safe_array(bounds_dict.get('y_max', None)),
                        u_min=safe_array(bounds_dict.get('u_min', None)),
                        u_max=safe_array(bounds_dict.get('u_max', None)),
                        du_min=safe_array(bounds_dict.get('du_min', None)),
                        du_max=safe_array(bounds_dict.get('du_max', None))
                    )

        mpc_params = mpc.MPCParams(dt=dt, N_horizon=n_horizon, weights=weights, bounds=bounds)
        
        if use_constraints:
            if 'standard' in ctrl_type:
                if mode == 'state_form':
                    return mpc.ConstrainedStandardStateForm(model, mpc_params, node_name=node_name, use_preview=use_preview, logger=logger)
                elif mode == 'output_form':
                    return mpc.ConstrainedStandardOutputForm(model, mpc_params, node_name=node_name, use_preview=use_preview, logger=logger)
            elif 'incremental' in ctrl_type:
                if mode == 'state_form':
                    return mpc.IncrementalStateForm(model, mpc_params, node_name=node_name, use_preview=use_preview, logger=logger)
                elif mode == 'output_form':
                    return mpc.IncrementalOutputForm(model, mpc_params, node_name=node_name, use_preview=use_preview, logger=logger)
            if 'integral' in ctrl_type:
                if mode == 'state_form':
                    int_limit = safe_array(params.get('int_limit'))
                    return mpc.ConstrainedIntegralStateForm(model, mpc_params, node_name=node_name, use_preview=use_preview, dt=dt, int_limit=int_limit, logger=logger)
                elif mode == 'output_form':
                    raise NotImplementedError("Integral Output Form MPC is not implemented yet.")
            else:
                raise ValueError(f"MPC requires mode 'state_form' or 'output_form', got: {mode}")
            
        else:
            if 'standard' in ctrl_type:
                if mode == 'state_form':
                    return mpc.UnconstrainedStateForm(model, mpc_params, use_preview=use_preview, logger=logger)
                elif mode == 'output_form':
                    return mpc.UnconstrainedOutputForm(model, mpc_params, use_preview=use_preview, logger=logger)
            if 'integral' in ctrl_type:
                if mode == 'state_form':
                    return mpc.UnconstrainedIntegralStateForm(model, mpc_params, use_preview=use_preview, logger=logger)
                elif mode == 'output_form':
                    raise NotImplementedError("Integral Output Form MPC is not implemented yet.")
            else:
                raise ValueError(f"MPC requires mode 'state_form', 'output_form', or 'integral_state_form', got: {mode}")

    raise ValueError(f"Unsupported controller type: {ctrl_type}")