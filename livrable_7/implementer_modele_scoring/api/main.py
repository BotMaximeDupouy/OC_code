import pandas as pd
from api.utils import load_model, load_data, get_client_data, client_id_in_data_test, get_sample_id, empty_data_response
from config import MAX_INTEREST_COLS

def get_features_importance(model,
                            features:list) -> pd.core.frame.DataFrame:
    '''get for each feature the importance for the model
    return it in DataFrame sorted deacreased'''
    importances = model.feature_importances_
    features_imp = pd.DataFrame(importances,
                                index=features,
                                columns=['feature_score']
                                )
    return features_imp.sort_values(by='feature_score',
                                    ascending=False)


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

def predict_solvability(row_data:pd.core.frame.DataFrame,
                        model)->str:
    '''predict client solvability'''
    res, proba = model.predict(row_data), model.predict_proba(row_data)
    return str(res[0]), str(proba[0])


def get_data_(params:str)->str:
    client_id = int(params) # ['client_id']
    data = load_data()
    response_data = empty_data_response()

    # essaye de retrouver la donnée pour le client_id donné
    index_not_found = client_id_in_data_test(data, int(client_id))

    if index_not_found:
        # TODO index ) false dans response
        response_data['error']['status'] = index_not_found
        response_data['error']['client_id_sample'] = str(get_sample_id(data))
        return response_data

    else :
        # TODO regarder importance de client_data
        client_data = get_client_data(data, client_id)
        data = data.drop(columns=['is_test'])
        model = load_model()
        features_imp = get_features_importance(model, client_data.columns)
        data_to_build = get_limit_values(features_imp, data, client_id)
        # top_features = features_imp.iloc[:MAX_INTEREST_COLS, :]
        response_data['error']['status'] = index_not_found
        response_data['data'] = data_to_build.to_json()
        return response_data

def make_prediction(client_data:dict):
    model = load_model()
    print('_________________')
    print(client_data)
    client_data = pd.DataFrame.from_dict(client_data, orient='index').transpose()
    print(client_data.head())
    prediction, probability = predict_solvability(client_data, model)
    return {"prediction" : prediction,
            "probabilies" : probability,}
