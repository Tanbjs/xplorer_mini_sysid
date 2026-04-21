import warnings
import os
import urllib3
import numpy as np
import mlflow

import logging
warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
sim_logger = logging.getLogger("Simulation")
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"


from xplorer_mini_sysid.lib.utils.mlflow import load_model

mlflow_uri = "https://mlflow.amarr.tan" 
mlflow.set_tracking_uri(mlflow_uri)
mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)

dmdc_model = load_model(client=mlflow_client, name='dmdc', version=1)
edmdc_model = load_model(client=mlflow_client, name='edmdc', version=1)

# load test trajectory
dmdc_run = mlflow_client.get_run(mlflow_client.get_model_version(name='dmdc', version=1).run_id)
edmdc_run = mlflow_client.get_run(mlflow_client.get_model_version(name='edmdc', version=1).run_id)

# check test trajectory is the same for both models
if dmdc_run.data.tags['test_traj'] != edmdc_run.data.tags