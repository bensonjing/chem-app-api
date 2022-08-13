from flask import Flask 

app = Flask(__name__) 

@app.get('/') 
def index_get(): 
  return 'Hello, World!'