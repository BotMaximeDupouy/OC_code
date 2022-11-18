import numpy as np
import pandas as pd
import seaborn as sns


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
