import json
import pandas as pd
import requests
import streamlit as st
from config import HOST, PORT

payload = {
    "client_id": None
}

## TITLE
st.title('Hello World')

## SIDE BAR INPUT CLIENT ID
client_id = st.sidebar.number_input('Insert client id',
                                            value=0)
payload["client_id"] = str(client_id)

## GET DATA
response = requests.post(''.join([HOST, PORT,'/get_data/']),
                         json=payload
                         ).json()

## CHECK STATUS
error = response["error"]["status"]

## IF INPUT ID ISNT KNOWN
if error :
    st.write("Cet identifiant ne correspond à aucun identifiant de client dans la base de donnée 😱")
    st.write("Exemples d'idenfiants connus :")
    st.write(', '.join([str(i) for i in eval(response["error"]["client_id_sample"])]))

## ELSE : LET THE MAGIE BEGGINS
else :
    # GET DF DATA FROM RESPONSE
    data = pd.DataFrame(eval(response["data"]))

    #GET CLIENT DATA
    client_value = data.loc[:, 'client_value']

    # INPUT HOWMANY FEATURE TO ANALYSED (N)
    number_features_to_eval = st.sidebar.slider(
        'Insert range of features you want to analyse',
        min_value=0,
        max_value=5, # crash si trop de colonnes
        value=0)

    ## ITERE SUR LES N FEATURES AYANT LE + D'IOMPACT
    for feature_name, data_feature in data.iloc[:number_features_to_eval,
                                                :].iterrows():
        # SI ON VEUT LE DESCRIPTIF DE LA FEATURE
        if st.sidebar.button(f'voir description de la colonne {feature_name}'):
            st.sidebar.write('cest la description')

        # CREER UN SIDEBAR POUR CHANGER INPUT DE LA FEATURE
        values = st.sidebar.slider(
        f'Move value for {feature_name}',
        min_value=float(data.loc[feature_name, 'min_value']),
        max_value=float(data.loc[feature_name, 'max_value']),
        value=float(data.loc[feature_name, 'client_value'])
        )
        data.loc[feature_name, 'client_value'] = values
        #st.write('Values:', values)

    prediction = requests.post(''.join([HOST, PORT,'/predict/']),
                               json=client_value.to_dict())
    st.write('prediction :', prediction.json())

    with st.expander("See explanation"):
        st.write("""
            The chart above shows some numbers I picked for you.
            I rolled actual dice for these, so they're *guaranteed* to
            be random.
        """)
        st.write("Ici")
