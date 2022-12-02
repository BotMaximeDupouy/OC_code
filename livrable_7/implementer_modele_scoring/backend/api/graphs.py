import io
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from api.feature_imp import FeaturesImportance

mapping_x_ticks = {
    '1.0': 'Default Client',
    '0.0': 'Not Default Client'
    }


def create_box_plot(data, feature:str, feature_value:float):
    fig, ax = plt.subplots(figsize=(12, 9))
    # create boxplot
    sns.boxplot(data=data[~data.TARGET.isna()],
                     y=feature,
                     x='TARGET',
                     orient="v",
                     showfliers = False,
                     palette=["#4286DE", "#EA365B"])
    # add client treshshold
    ax.axhline(feature_value,
               color='r',
               label='Client value')
    # add label and legend
    ax.legend()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = [mapping_x_ticks[i] for i in labels]
    ax.set_xticklabels(labels)
    ax.set_title(f'{feature}')
    ax.title.set_size(20)
    # create a buffer to store image data
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close()
    return img_buf


def create_force_plot(client_data, model, columns, prediction:int):
    plt.rcParams['figure.autolayout'] = True
    f_i = FeaturesImportance(model, columns)
    shap_values = f_i.get_shap_values(client_data)
    explainer = f_i.explainer
    shap.force_plot(np.around(explainer.expected_value[prediction], decimals = 2),
                          np.around(shap_values[prediction], decimals = 2),
                          np.around(client_data, decimals = 2),
                          matplotlib=True,
                          show=False,
                          text_rotation=15,
                          figsize=(20,6))
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close()
    return img_buf


def create_features_importance_model(model, columns):
    # create dict {columns_name:model_feature_importance}
    dict_f_i = dict(zip(columns, model.feature_importances_))
    # sorted by feature_importance
    dict_f_i = {k: v for k, v in sorted(dict_f_i.items(), key=lambda item: item[1], reverse=True)}
    # return barplot
    plt.rcParams['figure.autolayout'] = True
    plt.figure(figsize=(25, 14))
    matplotlib.rc('ytick', labelsize=15)
    matplotlib.rc('xtick', labelsize=15)
    sns.barplot(x=list(dict_f_i.values()), y=list(dict_f_i.keys()), orient='h', color='#4286DE')
    # create a buffer to store image data
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close()
    return img_buf
