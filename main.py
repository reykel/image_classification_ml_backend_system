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
        'message': 'Welcome sample'
        }

@app.post('/predict/image')
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split('.')[-1] in ('jpg', 'jpeg', 'png')
    if not extension:
        return 'Image must be jpg or png format!'
    image = components.read_imagefile(await file.read())
    data_response = components.prediction(image)

    return data_response


@app.get('/blog')
def index(limit=10, published: bool = True, sort: Optional[str] = None):
    # only get 10 published blogs
    if published:
        return {'data': f'{limit} published blogs from the db'}
    else:
        return {'data': f'{limit} blogs from the db'}


@app.get('/blog/unpublished')
def unpublished():
    return {'data': 'all unpublished blogs'}


@app.get('/blog/{id}')
def show(id: int):
    # fetch blog with id = id
    return {'data': id}


@app.get('/blog/{id}/comments')
def comments(id, limit=10):
    # fetch comments of blog with id = id
    return {'data': {'1', '2'}}


class Blog(BaseModel):
    title: str
    body: str
    published: Optional[bool]


@app.post('/blog')
def create_blog(blog: Blog):
    return {'data': f'Blog is created with title as {blog.title}'}
