from typing import Union

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
import uvicorn

from api.main import get_data_, make_prediction
from config import HOST, PORT
from models.pydantic_models import ClientId, Data



app = FastAPI()

@app.post('/get_data')
def get_data_for_client(payload: ClientId):
    payload = payload.dict()
    print(payload["client_id"])
    data = get_data_(payload["client_id"])
    return data

@app.post('/predict')
def predict_solvability(data: Data):
    data = jsonable_encoder(data)
    prediction = make_prediction(data)
    return prediction

if __name__ == '__main__':
    uvicorn.run(app, host=HOST, port=PORT)
