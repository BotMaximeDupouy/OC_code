import numpy as np
import pandas as pd


def client_id_in_data_test(data:pd.core.frame.DataFrame,
                           client_id:str
                           ) -> bool:
    '''check if client_id from input is on data_test'''
    if client_id in list(data[data["TARGET"].isna()].index):
        return False
    else:
        return True

def get_sample_id(data:pd.core.frame.DataFrame) -> list:
    '''return a list of 10 client_id in data_test'''
    index_ = list(data[data["TARGET"].isna()].index)
    return list(np.random.choice(index_, size=10))

def get_limit_values(features_imp:pd.core.frame.DataFrame,
                     data:pd.core.frame.DataFrame,
                     client_id:int
                     ) -> pd.core.frame.DataFrame:
    '''create a dataframe with for each feature :
    - min values
    - max values
    - client values
    '''
    max_values = []
    min_values = []
    client_values = []
    for col in list(features_imp.index):
        max_values.append(data[col].max())
        min_values.append(data[col].min())
        client_values.append(data.loc[data.index==client_id, col].values[0])
    features_imp['max_value'] = max_values
    features_imp['min_value'] = min_values
    features_imp['client_value'] = client_values
    return features_imp
