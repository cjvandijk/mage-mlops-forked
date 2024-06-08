import pandas as pd


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preps dataframe for downstream training task

    Args:
        df: The output from the upstream parent block
    
    Returns:
        df: transformed dataframe with reduced columns
    """

    # calculate target feature
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    # Eliminate outliers
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    # prepare features for training task
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    numerical = ['trip_distance', 'duration']

    # reduct memory load by removing unused columns
    df = df[categorical + numerical]
    
    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'