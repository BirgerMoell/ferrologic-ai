#!flask/bin/python
from flask import Flask

app = Flask(__name__)

from flask import request
from flask import jsonify
from keras.models import model_from_json

# loading the model
import json
import scipy.misc
@app.route('/')
def index():
    return "Hello, World!"

@app.route('/predict', methods=['POST'])
def predict():
    json_file = open('first_try.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("first_try.h5")
    print('loaded the model')
    print("request files is")
    print(request.files)
    file = request.files['image']
    testimage = scipy.misc.imresize(scipy.misc.imread(file),(150,150))
    print(testimage.shape)
    testimage = testimage.reshape((1,) + testimage.shape)
    prediction = loaded_model.predict(testimage).astype(float)
    print(prediction)
    return jsonify({ 'classification': { 'dog': prediction[0][0], 'cat' : 1-prediction[0][0]} })
if __name__ == '__main__':
    app.run(debug=True)
