import pandas as pd
from config import DATA_PATH

class DataTest:
    def __init__(self):
        pass

    def load_data():
        '''Load dataset with client_id and data'''
        data = pd.read_csv(DATA_PATH,
                           index_col="SK_ID_CURR")
        return data
