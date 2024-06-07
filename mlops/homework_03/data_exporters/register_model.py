import mlflow
from typing import Tuple
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from homework_03.utils.dump_file import dump_pickle

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(
    data: Tuple[DictVectorizer, LinearRegression], *args, **kwargs
    ) -> None:
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    EXPERIMENT_NAME = training_model_thru_mage
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)
    # experiment = mlflow.get_experiment("training_model_thru_mage")

    dv, lr = data
    dump_file((dv, lr), "models/lin_reg.bin")

    with mlflow.start_run():
        mlflow.log_artifact(dv, artifact_path="dictvectorizers")
        model_info = mlflow.sklearn.log_model(
            sk_model=lr, artifact_path="modelsxx"
        )

    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.rmse ASC"]
    )[0]
    
    # Register best model
    mlflow.register_model(model_uri=f"runs:/{best_run.info.run_id}/models", 
                          name="Taxi experiment linear regressio model"
                         )
