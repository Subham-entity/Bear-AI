import io
import torch
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from PIL import Image
from torchvision import transforms
from dataset.vocab import Vocabulary
from models.bear import Bear
from api.inference import generate_answer

app = FastAPI(
    title="Bear-VQA Transformer API",
    description="Visual Question Answering Engine using CNN + Transformer Cross-Attention"
)

# Global device and model configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "weights/bear_vqa_checkpoint.pt"  # Updated weights path for VQA model checkpoint

print("Loading model checkpoint...")
torch.serialization.add_safe_globals([Vocabulary])
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
vocab = checkpoint['vocab']
vocab_size = checkpoint['vocab_size']

# Initialize model architecture matching your training config
model = Bear(
    vocab_size=vocab_size,
    embed_dim=512,
    num_heads=8,
    hidden_dim=2048,
    num_layers=4
).to(DEVICE)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
def root():
    return {"status": "Bear VQA API is running!"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...), question: Optional[str] = Form(None)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file format.")
    
    tensor_image = transform(image)
    
    # Use default caption prompt if question is empty or blank
    prompt = question if question and question.strip() else "describe the image"
    
    result = generate_answer(model, vocab, tensor_image, question=prompt, device=DEVICE)
    
    return {
        "filename": file.filename,
        "question": prompt,
        "response": result
    }