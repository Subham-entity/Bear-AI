import io
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms
from dataset.vocab import Vocabulary
from models.bear import Bear

app = FastAPI(title="Bear-Transformer API", description="Image Captioning Engine using CNN + Transformer Cross-Attention")

# Global device and model configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "weights/visiotext_checkpoint.pt"  # Ensure your weights file is placed here locally

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
    # 1. Read uploaded image bytes
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 2. Preprocess image tensor
    tensor_image = transform(image)

    # 3. Autoregressive inference loop
    with torch.no_grad():
        encoder_features = model.encoder(tensor_image.unsqueeze(0).to(DEVICE))
        token_indices = [vocab.stoi["<START>"]]
        
        for _ in range(30):
            captions_tensor = torch.tensor(token_indices, dtype=torch.long).unsqueeze(0).to(DEVICE)
            logits = model.decoder(captions_tensor, encoder_features)
            predicted_token_id = logits[0, -1, :].argmax().item()
            
            if predicted_token_id == vocab.stoi["<END>"]:
                break
            token_indices.append(predicted_token_id)
            
        generated_words = [vocab.itos[idx] for idx in token_indices[1:]]
        caption = " ".join(generated_words)
        
    return {
        "filename": file.filename,
        "generated_caption": caption
    }