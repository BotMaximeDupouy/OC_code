from fastapi import FastAPI, Response, BackgroundTasks
import uvicorn

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
def predict_solvability(data_: Data):
    data_ = data_.dict()
    prediction = make_prediction(data_)
    return prediction


@app.get('/')
def get_img(graph_params: GraphParams, background_tasks: BackgroundTasks):
    graph_params = graph_params.dict()
    img_buf = make_butifuls_graphs(graph_params)
    background_tasks.add_task(img_buf.close)
    headers = {'Content-Disposition': 'inline; filename="out.png"'}
    return Response(img_buf.getvalue(), headers=headers, media_type='image/png')

if __name__ == '__main__':
    uvicorn.run(app, host=HOST, port=PORT)
