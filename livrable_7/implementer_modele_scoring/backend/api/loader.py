import pandas as pd
import pickle

from config import DATA_PATH, MODEL_PATH, MODEL_TO_USE

class Loader:
    def __init__(self):
        pass

    def load_data():
        '''Load dataset with client_id and data'''
        data = pd.read_csv(DATA_PATH,
                           index_col="SK_ID_CURR")
        return data

    def load_model():
        '''load the model used for prediction'''
        pickle_in = open(''.join([MODEL_PATH,
                                MODEL_TO_USE]),
                        "rb")
        return pickle.load(pickle_in)
