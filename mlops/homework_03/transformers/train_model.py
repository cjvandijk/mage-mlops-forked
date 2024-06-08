import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def start_train(df: pd.DataFrame) -> tuple[DictVectorizer, LinearRegression]:
    """
    Trains a linear regression model. 
    Prints the intercept for homework answer.

    Args:
        df: output from the upstream parent block
    

    Returns:
        dv: the DictVectorizer made from the train_df data
        lr: the linear regression model
    """


    y_train = df['duration'].values
    training_cols = ['PULocationID', 'DOLocationID', 'trip_distance']
    df_train = df[training_cols]
    
    # create matrix for linear regressor
    train_dicts = df_train.to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)

    # train regressor
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print("Intercept:",lr.intercept_)

    return dv, lr


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'