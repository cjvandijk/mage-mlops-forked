from pathlib import Path
import pickle
import tempfile
from typing import Tuple

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(
    data: Tuple[DictVectorizer, LinearRegression], *args, **kwargs
    ) -> None:
    """
    Logs the dictvectorizer to mlflow artifacts and the linear 
    regression model to mlflow models. Registers the logged model 
    in mlflow.

    Args:
        data: The output from the upstream parent block containing a tuple
        with a dictvectorizer and a linear regression model

    Output:
        No output is returned
    """


    EXPERIMENT_NAME = "training_model_thru_mage"

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    dv, lr = data

    # save model and dictvectorizer to temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "lin_reg.bin"
        with open(path, "wb") as f_out:
            pickle.dump((dv, lr), f_out)

        # log model and dv artifact
        with mlflow.start_run():
            mlflow.log_artifact(path)
            model_info = mlflow.sklearn.log_model(
                sk_model=lr, artifact_path="modelsxx"
            )

    # register model in mlflow
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        # order_by=["metrics.rmse ASC"]
    )[0]
    
    # Register best model
    mlflow.register_model(model_uri=f"runs:/{best_run.info.run_id}/models", 
                          name="Taxi experiment linear regressio model"
                         )
