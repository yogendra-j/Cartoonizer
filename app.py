import os
from flask import Flask, redirect, url_for, request, render_template
from catoonizer import *


def createapp():
    return Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'static', 'uploads',
                                 "xx." + f.filename.split('.')[-1])

        f.save(file_path)
        out = cv2.imread(file_path)
        out = makecartoon(out)
        cv2.imwrite(
            os.path.join(basepath, 'static', 'uploads',
                         "xxx." + f.filename.split('.')[-1]), out)
        return redirect(url_for('static',
                                filename='uploads/' + "xxx." +
                                f.filename.split('.')[-1]),
                        code=301)


if __name__ == '__main__':
    app = createapp()
    app.run(debug=True)
