from flask import Flask, request

app = Flask(__name__) 

@app.get('/') 
def index_get(): 
  return {
    "message": "Hello, World!"
  }

@app.post('/pic')
def pic_post(): 
  file = request.files['file']
  return {"message": "Pic Post Success"}