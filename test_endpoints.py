from fastapi import FastAPI
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_home():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {
        'message': 'Dispositivo de control fitosanitario running'
        }

def test_get_prediction():
    filename = './files/test.jpg'
    response = client.post('/predict/image',
                           files={'file': (
                               'filename', open(filename, 'rb'),
                               'image/jpeg'
                               )}
                           )
    assert response.status_code == 200
