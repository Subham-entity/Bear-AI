import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from dataset.vocab import Vocabulary
from dataset.dataloader import FlickrDataset, CollateFn
from models.bear import Bear

def train():
    # 1. Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Dataset from Hugging Face
    print("Loading Flickr8k dataset...")
    hf_dataset = load_dataset("jxie/flickr8k", split="train")

    # 3. Build Vocabulary
    print("Building vocabulary...")
    all_captions = []
    for sample in hf_dataset:
        for i in range(5):
            all_captions.append(sample[f"caption_{i}"])
            
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(all_captions)
    vocab_size = len(vocab)
    print(f"Vocabulary built! Total unique tokens: {vocab_size}")
    
    # 4. Image Transforms & DataLoader Setup
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = FlickrDataset(hf_dataset, vocab, transform=transform)
    collate_fn = CollateFn(pad_idx=vocab.stoi["<PAD>"])
    
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=2
    )

    # 5. Initialize Model, Loss, and Optimizer
    embed_dim = 256
    num_heads = 8
    hidden_dim = 512
    num_layers = 2

    model = Bear(vocab_size, embed_dim, num_heads, hidden_dim, num_layers).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # 6. Training Loop (Running for 3 initial epochs to test)
    num_epochs = 3
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (images, captions) in enumerate(dataloader):
            images = images.to(device)
            captions = captions.to(device)

            # Teacher forcing splits
            decoder_inputs = captions[:, :-1]
            targets = captions[:, 1:]

            # Forward pass
            optimizer.zero_grad()
            logits = model(images, decoder_inputs)

            # Compute loss
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            
            # Backward pass & optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"--> Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")

    # Save Checkpoint to Google Drive
    checkpoint_path = "/content/drive/MyDrive/visiotext_checkpoint.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'vocab_size': vocab_size
    }, checkpoint_path)
    print(f"Model checkpoint saved successfully to {checkpoint_path}!")

if __name__ == "__main__":
    train()