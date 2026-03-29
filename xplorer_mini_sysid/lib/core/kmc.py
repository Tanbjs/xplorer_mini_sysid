from abc import ABC, abstractmethod

from kmc.utils.model_wrapper import DMDcWrapper, EDMDcWrapper, DeepModelWrapper

from .model import LinearModel, KoopmanModel


class BaseKMC(ABC):
    """
    Base class for Koopman-based controllers.

    Attributes:
        model: KoopmanModel - The Koopman model of the system, containing the system dynamics and lifting function.

    Methods:
        __init__(model_wrapper): Initializes the KMC with a given model wrapper, extracting the Koopman model.
        __get_model(model_wrapper): Internal method to convert a model wrapper into a KoopmanModel instance, including the lifting function and any scalers.
    """

    model: KoopmanModel

    def __init__(self, 
                 model_wrapper: DMDcWrapper | EDMDcWrapper | DeepModelWrapper):
        self.model = self.__get_model(model_wrapper)

    def __get_model(self, model_wrapper: DMDcWrapper | EDMDcWrapper | DeepModelWrapper):

        if isinstance(model_wrapper, DMDcWrapper):
            lift_func = lambda x: x 
        elif isinstance(model_wrapper, EDMDcWrapper):
            lift_func = lambda x: model_wrapper.model._obs_func.transform(x)
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
