from flask import Flask, request, redirect, url_for

import result


app = Flask(__name__)


count = 0


@app.route('/')
def index_get():
    return 'hello'
    # return redirect(url_for('pic'))


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


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)
