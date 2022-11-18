import numpy as np
import pandas as pd
import shap


class FeaturesImportance:
    def __init__(self, model, columns):
        self.model = model
        self.explainer = shap.TreeExplainer(model)
        self.columns = columns

    def get_shap_values(self, client_data):
        shap_values = self.explainer.shap_values(client_data)
        return shap_values

    def get_feature_importance(self, shap_values:np.array):
        vals = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(
            list(zip(self.columns, sum(vals))),
            columns=['col_name','feature_importance_vals']
            )
        feature_importance.sort_values(by=['feature_importance_vals'],
                                       ascending=False,
                                       inplace=True)
        return feature_importance.set_index('col_name', drop=True)
