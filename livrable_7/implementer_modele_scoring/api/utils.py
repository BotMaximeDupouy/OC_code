import numpy as np
import pandas as pd
import pickle
from api.data import DataTest
from config import MODEL_PATH, MODEL_TO_USE


def load_model():
    '''load the model used for prediction'''
    pickle_in = open(''.join([MODEL_PATH,
                              MODEL_TO_USE]),
                    "rb")
    return pickle.load(pickle_in)

def load_data()->pd.core.frame.DataFrame:
    '''load data'''
    return DataTest.load_data()

def get_client_data(data, client_id:str)->pd.core.frame.DataFrame:
    return data[data.index==int(client_id)].drop(columns=['is_test'])

def client_id_in_data_test(data:pd.core.frame.DataFrame,
                           client_id:str
                           ) -> bool:
    '''check if client_id from input is on data_test'''
    if client_id in list(data[data["is_test"]==1].index):
        return False
    else:
        return True

def get_sample_id(data:pd.core.frame.DataFrame) -> list:
    '''return a list of 10 client_id in data_test'''
    index_ = list(data[data["is_test"]==1].index)
    return list(np.random.choice(index_, size=10))

def empty_data_response():
    return  {
        'error': {'status' : None,
                   'client_id_sample' : None,
                   },
        'data': None,
    }
