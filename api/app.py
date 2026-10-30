import io
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms
from dataset.vocab import Vocabulary
from models.bear import Bear
from api.inference import generate_caption

app = FastAPI(title="Bear-Transformer API", description="Image Captioning Engine using CNN + Transformer Cross-Attention")

# Global device and model configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "weights/bear_checkpoint.pt"  # Ensure your weights file is placed here locally

print("Loading model checkpoint...")
torch.serialization.add_safe_globals([Vocabulary])
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
vocab = checkpoint['vocab']
vocab_size = checkpoint['vocab_size']

# Initialize model architecture and load saved weights
model = Bear(vocab_size, embed_dim=256, num_heads=8, hidden_dim=512, num_layers=2).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Standard ImageNet transform pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
def root():
    return {"status": "Bear API is running!"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor_image = transform(image)

    # Calls the inference engine cleanly
    caption = generate_caption(model, vocab, tensor_image, device=DEVICE)
    
    return {
        "filename": file.filename,
        "generated_caption": caption
    }