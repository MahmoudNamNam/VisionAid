import os
from io import BytesIO
import requests
import torch
from torchvision import models, transforms
from PIL import Image

_model = None
_classes = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _download_file(url: str, filename: str):
    if os.path.exists(filename):
        return

    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")
    except requests.RequestException as e:
        if os.path.exists(filename):
            os.remove(filename)
        raise RuntimeError(f"Download failed for {filename}: {e}")


def load_labels(file_name="categories_places365.txt"):
    global _classes
    if _classes is not None:
        return _classes

    label_url = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"
    _download_file(label_url, file_name)

    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            _classes = [line.strip().split(' ')[0][3:] for line in f if line.strip()]
        print(f"Loaded {len(_classes)} scene categories.")
    except Exception as e:
        raise RuntimeError(f"Failed to read label file {file_name}: {e}")
    
    return _classes


def load_scene_model(weight_file='resnet50_places365.pth.tar'):
    global _model
    if _model is not None:
        return _model

    model_url = 'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
    _download_file(model_url, weight_file)

    try:
        model = models.resnet50(weights=None, num_classes=365)
        checkpoint = torch.load(weight_file, map_location='cpu')
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        _model = model
        print("ResNet50 Places365 model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    return _model


def preprocess_image(image_bytes: bytes):
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return transform(img).unsqueeze(0).to(DEVICE)
    except Exception as e:
        raise RuntimeError(f"Image preprocessing failed: {e}")


def classify_scene(image_bytes: bytes, top_n: int = 5):
    try:
        model = load_scene_model()
        classes = load_labels()
        if model is None or classes is None:
            raise RuntimeError("Model or labels not properly loaded.")

        input_tensor = preprocess_image(image_bytes)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            top_prob, top_catid = torch.topk(probs, top_n)

        return [
            (classes[catid], round(prob.item(), 4))
            for prob, catid in zip(top_prob[0], top_catid[0])
        ]

    except Exception as e:
        raise RuntimeError(f"Scene classification failed: {e}")
