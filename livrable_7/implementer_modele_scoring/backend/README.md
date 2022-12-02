API used for "implementer un modele de scoring"

Install requirements.txt :
- make install
- or pip install -r requirements.txt -r requirements-dev.txt

Execute tests :
- make tests
- or pytest ./tests.test.py

Run app :
- unicorn app:app

Endpoints description :
- POST(/get_data):
  payload : client_id
  return client data and min/max value for each column

- POST(/predict):
  payload : client data
  return prediction

- GET(/):
  get graph
