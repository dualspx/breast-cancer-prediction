import os
from flask import Flask, render_template, request, jsonify
import os
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch
from flask import send_file

# Specify the template folder explicitly
template_dir = os.path.abspath('pages')
app = Flask(__name__, template_folder=template_dir)


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(300),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_tensor, model):
    # Load the ResNet-34 model
    model_path = '../dataset/models'
    if os.path.exists(model_path):
        #initialize model
        print(model)
        if model == 'resnet34':
            model = models.resnet34(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 1)
            model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth')), strict=False)
            model.eval()
        elif model == 'resnet50':
            model = models.resnet50(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 1)
            model.load_state_dict(torch.load(os.path.join(model_path, 'resnet50.pth')), strict=False)
            model.eval()
        else:
            return 'Model not found'
        # Make prediction
        with torch.no_grad():
            image_pil = transforms.ToPILImage()(image_tensor)
            image_pil = image_pil.convert('RGB')
            image_tensor = transforms.ToTensor()(image_pil)
            image_tensor = image_tensor.unsqueeze(0)
            prediction = model(image_tensor)
            predicted = torch.sigmoid(prediction).item()  # Apply sigmoid to get probability
            predicted_class = 1 if predicted > 0.5 else 0  # Convert probability to class
            return predicted_class
    else:
        print('Model not found')
    # Convert image to RGB mode
    # image_tensor = image_tensor.convert('RGB')
    

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
        return jsonify({'error': 'No file part'})
    
    file_image = request.files['image']
    
    if file_image.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file_image:
        model = request.form['model']
        image_bytes = file_image.read()
        image = Image.open(file_image)
        image_tensor = transforms.ToTensor()(image)
        prediction = get_prediction(image_tensor, model)
        print(prediction)

        prediction_text = 'Malignant' if prediction == 1 else 'Benign'
        
        return jsonify(
            {
            'model': request.form['model'],
            'prediction': str(prediction_text),
            'prediction_value': str(prediction),
            'file_name': str(file_image.filename),
            'file_size': str(len(image_bytes)),
            'status': 'OK'
            }
        ), 200, {'Content-Type': 'application/json'}
    

@app.route('/result', methods=['GET'])
def result():
    return 'Result page'

if __name__ == '__main__':
    app.run()