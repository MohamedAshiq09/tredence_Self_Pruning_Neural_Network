from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.network import SelfPruningNet

app = FastAPI(title="Self-Pruning Neural Network API")

# Global model variable
model = None
device = None
transform = None

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']


@app.on_event("startup")
async def load_model():
    """Load trained model on startup"""
    global model, device, transform
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SelfPruningNet().to(device)
    
    # Try to load best model
    model_paths = [
        'results/lambda_0.001/best_model_lambda_0.001.pth',
        'results/lambda_0.0001/best_model_lambda_0.0001.pth',
        'results/lambda_0.01/best_model_lambda_0.01.pth'
    ]
    
    loaded = False
    for path in model_paths:
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            loaded = True
            print(f"Loaded model from {path}")
            break
    
    if not loaded:
        print("Warning: No trained model found. Using untrained model.")
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


@app.get("/")
async def root():
    return {"message": "Self-Pruning Neural Network API", "status": "running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict class for uploaded image
    
    Args:
        file: Image file (JPEG, PNG)
        
    Returns:
        Prediction results with class probabilities
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = probabilities.argmax().item()
            confidence = probabilities[predicted_class].item()
        
        # Get top 3 predictions
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        top3_predictions = [
            {"class": CIFAR10_CLASSES[idx], "probability": prob.item()}
            for idx, prob in zip(top3_idx, top3_prob)
        ]
        
        return {
            "predicted_class": CIFAR10_CLASSES[predicted_class],
            "confidence": confidence,
            "top3_predictions": top3_predictions
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.get("/model-stats")
async def get_model_stats():
    """
    Get model statistics including sparsity information
    
    Returns:
        Model statistics and sparsity metrics
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        sparsity_stats = model.get_overall_sparsity()
        
        return {
            "overall_sparsity_percent": sparsity_stats['overall_sparsity'],
            "total_weights": sparsity_stats['total_weights'],
            "pruned_weights": sparsity_stats['pruned_weights'],
            "active_weights": sparsity_stats['total_weights'] - sparsity_stats['pruned_weights'],
            "layer_statistics": sparsity_stats['layer_stats'],
            "device": str(device)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
