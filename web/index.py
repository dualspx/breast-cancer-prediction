import os
from flask import Flask, render_template, request
import os

import torch

# Specify the template folder explicitly
template_dir = os.path.abspath('pages')
app = Flask(__name__, template_folder=template_dir)

# create a route for the web app on pages/index.html
@app.route('/')
def index():
    return render_template('index.html')

@app.get('/hello')
def hello():
    return 'Hello, World!'

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'No file uploaded'
    else:
        file = request.files['image']
        return trainModel(file)
        # return file.filename

@app.route('/result', methods=['GET'])
def result():
    return 'Result page'

def trainModel(file):
    model_path = '../dataset/models'
    if os.path.exists(model_path):
        model_file = os.path.join(model_path, 'resnet34.pth')
        model = torch.load(model_file)
        return 'Model trained'
    else:
        return 'Model not found'
    # Rest of the code for training the model
    
if __name__ == '__main__':
    app.run()