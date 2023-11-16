from fastapi import FastAPI, Depends, File, UploadFile
from typing import Optional
from pydantic import BaseModel

from pydantic_settings import BaseSettings
import components
from functools import lru_cache

app = FastAPI()

class Settings(BaseSettings):
    debug: bool = False

    class Config:
        env_file = '.env'

@lru_cache
def get_settings():
    return Settings()

settings = get_settings()
DEBUG = settings.debug

app = FastAPI()

print(DEBUG)
@app.get('/')
def root(settings:Settings = Depends(get_settings)):
    return {
        'message': 'Dispositivo de control fitosanitario running'
        }

@app.post('/predict/image')
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split('.')[-1] in ('jpg', 'jpeg', 'png')
    if not extension:
        return 'Image must be jpg or png format!'
    image = components.read_imagefile(await file.read())
    data_response = components.prediction(image)

    return data_response
