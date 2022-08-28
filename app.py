from flask import Flask, request

import result


app = Flask(__name__)


count = 0


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


if __name__ == '__main__':
    app.run(debug=True, port=5000)
