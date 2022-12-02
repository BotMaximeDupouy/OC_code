import pytest

from api.utils import client_id_in_data_test, get_sample_id
from api.main import make_prediction
from api.loader import Loader



data = Loader.load_data()



@pytest.mark.parametrize(
    "input_client_id, is_in_test",
    [pytest.param(1, True, id="test id =1"),
     pytest.param(323894, False, id='id = 323894')
    ]
    )
def test_id_not_in_data_test(input_client_id, is_in_test):
    response = client_id_in_data_test(data, input_client_id)
    assert response == is_in_test


def test_get_sample_id():
    response = get_sample_id(data)
    assert len(response) == 10


@pytest.mark.parametrize(
    "input_data_client, expected_prediction",
    [
        pytest.param(
            {
                'PREV_MEAN_MIN_AMT_PAYMENT': 1479.99,
                'EXT_SOURCE_2': 0.6160995323,
                'AMT_GOODS_PRICE': 886500.0,
                'EXT_SOURCE_3': 0.5108529062,
                'PREV_SUM_MIN_AMT_PAYMENT': 4439.97,
                'DAYS_BIRTH': -12199.0,
                'DAYS_EMPLOYED': -494.0,
                'EXT_SOURCE_1': 0.5021298057,
                'BUREAU_MAX_DAYS_CREDIT_ENDDATE': 4433.041953435,
                'OWN_CAR_AGE': 6.0,
                'DAYS_ID_PUBLISH': -4143.0,
                'AMT_CREDIT': 1079581.5,
                'BUREAU_MAX_DAYS_CREDIT': -489.2978166237,
                'AMT_ANNUITY': 55251.0,
                'BUREAU_MAX_DAYS_ENDDATE_FACT': -526.9859477762
                },
            1,
            id="Client 323894 -> prediction 1"
            ),
        pytest.param(
            {
            'EXT_SOURCE_1': 0.502129805655716,
            'EXT_SOURCE_2': 0.616683816204787,
            'EXT_SOURCE_3': 0.8277026681625442,
            'DAYS_BIRTH': -21785,
            'AMT_CREDIT': 172021.5,
            'AMT_ANNUITY': 13441.5,
            'DAYS_EMPLOYED': -6822,
            'AMT_GOODS_PRICE': 148500.0,
            'DAYS_ID_PUBLISH': -3487,
            'OWN_CAR_AGE': 12.06109081865161,
            'BUREAU_MAX_DAYS_CREDIT': -653.0,
            'BUREAU_MAX_DAYS_CREDIT_ENDDATE': 78.0,
            'BUREAU_MAX_DAYS_ENDDATE_FACT': -255.0,
            'PREV_SUM_MIN_AMT_PAYMENT': 40975.648490716536,
            'PREV_MEAN_MIN_AMT_PAYMENT': 13438.182243201416
            },
         0,
         id="Client 417980 -> prediction 0")
     ]
    )
def test_prediction(input_data_client, expected_prediction):
    response = make_prediction(input_data_client)
    assert int(response['prediction']) == expected_prediction
