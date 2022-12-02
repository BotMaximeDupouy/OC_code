import pandas as pd

from api.feature_imp import FeaturesImportance
from api.graphs import (create_features_importance_model,
                        create_box_plot,
                        create_force_plot)
from api.loader import Loader
from api.utils import client_id_in_data_test, get_sample_id, get_limit_values
from config import EMPTY_RESPONSE, COLUMNS_ORDER


# def get_features_importance(model,
#                             features:list) -> pd.core.frame.DataFrame:
#     '''get for each feature the importance for the model
#     return it in DataFrame sorted deacreased'''
#     importances = model.feature_importances_
#     features_imp = pd.DataFrame(importances,
#                                 index=features,
#                                 columns=['feature_score']
#                                 )
#     return features_imp.sort_values(by='feature_score',
#                                     ascending=False)




def predict_solvability(row_data:pd.core.frame.DataFrame,
                        model)->str:
    '''predict client solvability'''
    res, proba = model.predict(row_data), model.predict_proba(row_data)
    return str(res[0]), str(max(proba[0]))


def get_data_(params:str)->str:
    client_id = int(params)
    data = Loader.load_data()


    # essaye de retrouver la donnée pour le client_id donné
    index_not_found = client_id_in_data_test(data, int(client_id))

    if index_not_found:
        response_data = EMPTY_RESPONSE
        response_data['error']['status'] = index_not_found
        response_data['error']['client_id_sample'] = str(get_sample_id(data))

    else :
        response_data = EMPTY_RESPONSE
        data = data.drop(columns=['TARGET'])
        model = Loader.load_model()

        # calculte client importance feature with shap
        f_i = FeaturesImportance(model, data.columns)
        shap_values = f_i.get_shap_values(data[data.index==client_id])
        features_imp = f_i.get_feature_importance(shap_values)
        data_to_build = get_limit_values(features_imp, data, client_id)

        response_data['error']['status'] = index_not_found
        response_data['data'] = data_to_build.to_json()
    return response_data

def make_prediction(client_data:dict) -> dict:
    '''take input client data and return prediction and probabilitie'''
    model = Loader.load_model()
    client_data = pd.DataFrame.from_dict(
        client_data, orient='index').transpose()[COLUMNS_ORDER]
    prediction, probability = predict_solvability(client_data, model)
    print(prediction, probability)
    return {"prediction" : prediction,
            "probabilies" : probability,}

def make_butifuls_graphs(graph_params):

    data = Loader.load_data()
    model = Loader.load_model()

    if graph_params['which_graph'] == 'features_importance_model':
        return create_features_importance_model(model,
                                                data.columns.drop('TARGET')
                                                )

    if graph_params['which_graph'] == 'client_distribution':
        feature_client_value = float(
            graph_params['data_client'][graph_params['feature']]
            )
        return create_box_plot(data,
                               graph_params['feature'],
                               feature_client_value)

    if graph_params['which_graph'] == 'force_plot':
        client_data = pd.DataFrame.from_dict(
            graph_params['data_client'], orient='index'
            ).transpose()[COLUMNS_ORDER]
        return create_force_plot(client_data,
                                 model,
                                 data.columns.drop('TARGET'),
                                 int(graph_params['prediction'])
                                )
