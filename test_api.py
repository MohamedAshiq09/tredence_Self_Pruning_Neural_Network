import requests
import json
from PIL import Image
import numpy as np


def test_health():
    """Test health endpoint"""
    response = requests.get("http://localhost:8000/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    print()


def test_model_stats():
    """Test model stats endpoint"""
    response = requests.get("http://localhost:8000/model-stats")
    print("Model Statistics:")
    print(json.dumps(response.json(), indent=2))
    print()


def test_predict():
    """Test prediction with a sample image"""
    img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    img.save('test_image.png')
    
    with open('test_image.png', 'rb') as f:
        files = {'file': ('test_image.png', f, 'image/png')}
        response = requests.post("http://localhost:8000/predict", files=files)
    
    print("Prediction Result:")
    print(json.dumps(response.json(), indent=2))
    print()


if __name__ == '__main__':
    print("Testing Self-Pruning Neural Network API\n")
    print("="*50)
    
    try:
        test_health()
        test_model_stats()
        test_predict()
        print("All tests passed!")
    except Exception as e:
        print(f"Error: {e}")
