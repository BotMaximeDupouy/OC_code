from typing import Union

from fastapi import FastAPI, Response, BackgroundTasks
import uvicorn

import io
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

from api.main import get_data_, make_prediction, make_butifuls_graphs
from config import HOST, PORT
from models.pydantic_models import ClientId, Data, GraphParams



app = FastAPI()

@app.post('/get_data')
def get_data_for_client(payload: ClientId):
    payload = payload.dict()
    data = get_data_(payload["client_id"])
    return data

@app.post('/predict')
def predict_solvability(data: Data):
    data = data.dict()
    prediction = make_prediction(data)
    return prediction


@app.get('/')
def get_img(graph_params: GraphParams, background_tasks: BackgroundTasks):
    graph_params = graph_params.dict()
    img_buf = make_butifuls_graphs(graph_params)
    background_tasks.add_task(img_buf.close)
    headers = {'Content-Disposition': 'inline; filename="out.png"'}
    return Response(img_buf.getvalue(), headers=headers, media_type='image/png')


# @app.get('/client_distribution')
# def get_img(background_tasks: BackgroundTasks, payload: ClientId):
#     payload = payload.dict()
#     img_buf = make_butifuls_graphs('client_distribution')
#     background_tasks.add_task(img_buf.close)
#     headers = {'Content-Disposition': 'inline; filename="out_2.png"'}
#     return Response(img_buf.getvalue(), headers=headers, media_type='image/png')
# TODO : get feature to evaluate woth boxplot



if __name__ == '__main__':
    uvicorn.run(app, host=HOST, port=PORT)
