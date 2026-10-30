import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

from dataset.vocab import Vocabulary
from dataset.dataloader import FlickrDataset, CollateFn
from models.bear import Bear

def validate(model, dataloader, criterion, device, vocab_size):
    model.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for images, captions in dataloader:
            images = images.to(device)
            captions = captions.to(device)

            # Teacher forcing splits
            decoder_inputs = captions[:, :-1]
            targets = captions[:, 1:]

            # Forward pass
            logits = model(images, decoder_inputs)

            # Compute loss
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(dataloader)
    return avg_val_loss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Dataset and split into Train & Validation (e.g., 90% train, 10% validation)
    print("Loading Flickr8k dataset...")
    hf_dataset = load_dataset("jxie/flickr8k", split="train")
    split_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42)
    train_hf = split_dataset["train"]
    val_hf = split_dataset["test"]

    # Build Vocabulary using training captions
    print("Building vocabulary...")
    all_captions = []
    for sample in train_hf:
        for i in range(5):
            all_captions.append(sample[f"caption_{i}"])
            
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(all_captions)
    vocab_size = len(vocab)
    print(f"Vocabulary built! Total unique tokens: {vocab_size}")

    # Transforms & Datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FlickrDataset(train_hf, vocab, transform=transform)
    val_dataset = FlickrDataset(val_hf, vocab, transform=transform)
    
    collate_fn = CollateFn(pad_idx=vocab.stoi["<PAD>"])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # Initialize Model, Loss, Optimizer
    embed_dim = 256
    num_heads = 8
    hidden_dim = 512
    num_layers = 2

    model = Bear(vocab_size, embed_dim, num_heads, hidden_dim, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    num_epochs = 15
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # --- TRAINING ---
        model.train()
        total_train_loss = 0
        for batch_idx, (images, captions) in enumerate(train_loader):
            images, captions = images.to(device), captions.to(device)

            decoder_inputs = captions[:, :-1]
            targets = captions[:, 1:]

            optimizer.zero_grad()
            logits = model(images, decoder_inputs)
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- EVALUATION ---
        avg_val_loss = validate(model, val_loader, criterion, device, vocab_size)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save best checkpoint based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = "/content/drive/MyDrive/bear_checkpoint.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'vocab_size': vocab_size
            }, checkpoint_path)
            print(f"Model checkpoint saved successfully to {checkpoint_path}!")

if __name__ == "__main__":
    train()