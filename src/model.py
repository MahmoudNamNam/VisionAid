# model.py (partial for SCENE_CONTEXTS)


import torch
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
import os
import requests

def _download_file(url: str, filename: str):
    if not os.path.exists(filename):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

def load_labels(file_name="categories_places365.txt"):
    _download_file("https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt", file_name)
    with open(file_name) as class_file:
        classes = [line.strip().split(' ')[0][3:] for line in class_file]
    return classes

def load_scene_model(weight_file='resnet50_places365.pth.tar'):
    model = models.resnet50(num_classes=365)
    model_url = 'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
    _download_file(model_url, weight_file)
    checkpoint = torch.load(weight_file, map_location=torch.device('cpu'))
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)

def classify_scene(image_bytes):
    model = load_scene_model()
    labels = load_labels()
    input_tensor = preprocess_image(image_bytes)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, 3)
    return [(labels[top_indices[0][i]], top_probs[0][i].item()) for i in range(top_probs.size(1))]
