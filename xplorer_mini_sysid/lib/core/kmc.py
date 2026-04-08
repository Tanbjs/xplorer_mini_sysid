from abc import ABC, abstractmethod

from kmc.utils.model_wrapper import DMDcWrapper, EDMDcWrapper, DeepModelWrapper

from .model import LinearModel, KoopmanModel


class BaseKMC(ABC):
    model: KoopmanModel

    def __init__(self, 
                 model_wrapper: DMDcWrapper | EDMDcWrapper | DeepModelWrapper):
        self.model = self.__get_model(model_wrapper)

    def __edmdc_transform(self, model_wrapper, x):
        """
        Handles transformation for EDMDc with dimensionality check.
        - Input 1D (vector): Reshape to 2D, transform, then flatten back to 1D.
        - Input 2D (batch): Transform directly.
        """
        if x.ndim == 1:
            return model_wrapper.model._obs_func.transform(x.reshape(1, -1)).flatten()
        return model_wrapper.model._obs_func.transform(x)

    def __get_model(self, model_wrapper: DMDcWrapper | EDMDcWrapper | DeepModelWrapper):
        
        if isinstance(model_wrapper, DMDcWrapper):
            lift_func = lambda x: x 
            
        elif isinstance(model_wrapper, EDMDcWrapper):
            lift_func = lambda x: self.__edmdc_transform(model_wrapper, x)
            
        elif isinstance(model_wrapper, DeepModelWrapper):
            lift_func = lambda x: model_wrapper.model.model.lift(x).detach().cpu().numpy()
            
        else:
            raise ValueError(f"Unsupported model wrapper type: {type(model_wrapper)}")

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
            scaler_y=getattr(model_wrapper, 'scaler_y', None)
        )