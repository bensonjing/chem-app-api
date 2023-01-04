from flask import Flask, request

import result
import os

os.environ["PYTHON_VERSION"] = "3.10.6"


app = Flask(__name__)


count = 0


@app.route('/')
def index_get():
    return 'hello'


@app.post('/pic')
def pic_post():
    global count

    file = request.files['file']
    file.save('photo/userPhoto.jpg')

    if count == 0:
        result.train()
        count += 1

    resultInt = result.getResult()
    return {"result": resultInt}
