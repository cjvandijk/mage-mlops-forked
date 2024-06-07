import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    """
    Loads March 2023 data from NYC yellow taxi trip data.

    Returns:
        pd.DataFrame
    """

    file_loc = "mlops/homework_03/taxi_data/yellow_tripdata_2023-03.parquet"

    return pd.read_parquet(file_loc)


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'