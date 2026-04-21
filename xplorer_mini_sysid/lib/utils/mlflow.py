import mlflow
from xplorer_mini_sysid.lib.utils.controller import Wrapper 


def load_model(client: mlflow.MlflowClient, 
               name, 
               version) -> Wrapper:
    
    try:
        model = client.get_model_version(name, version)
        run = client.get_run(model.run_id)
        loaded_model = mlflow.pyfunc.load_model(model.source).unwrap_python_model()
        print(f"Successfully loaded model '{name}' from {run.info.run_name}.")
        return loaded_model

    except Exception as e:
        print(f"Failed to load model: {e}")