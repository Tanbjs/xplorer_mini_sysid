import mlflow
from xplorer_mini_sysid.lib.utils.controller import Wrapper 


def load_model(client: mlflow.MlflowClient, name, version) -> Wrapper:
    try:
        model = client.get_model_version(name, version)
        model_uri = f"models:/{name}/{version}"
        loaded_model = mlflow.pyfunc.load_model(model_uri).unwrap_python_model()
        return loaded_model
    except Exception as e:
        raise RuntimeError(f"Model Loading Failed: {name} v{version}. Error: {e}")