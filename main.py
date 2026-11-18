import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from dataset.vocab import Vocabulary
from dataset.dataloader import FlickrDataset, CollateFn
from models.bear import Bear
import math

def validate(model, dataloader, criterion, device, vocab_size, pad_idx):
    model.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for images, full_seqs, answer_start_idxs in dataloader:
            images = images.to(device)
            full_seqs = full_seqs.to(device)

            # Teacher forcing splits
            decoder_inputs = full_seqs[:, :-1]
            targets = full_seqs[:, 1:]

            # Construct targets mask (-100 for question/padding, active token indices for answers)
            targets_mask = targets.clone()
            for i, start_idx in enumerate(answer_start_idxs):
                ans_start_in_target = max(0, start_idx.item() - 1)
                targets_mask[i, :ans_start_in_target] = -100
                targets_mask[i, targets[i] == pad_idx] = -100

            # Forward pass
            logits = model(images, decoder_inputs)

            # Compute loss
            loss = criterion(logits.reshape(-1, vocab_size), targets_mask.reshape(-1))
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(dataloader)
    return avg_val_loss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Dataset and split into Train & Validation (e.g., 90% train, 10% validation)
    print("Loading VQA dataset...")
    hf_dataset = load_dataset("merve/vqav2-small", split="validation")  # ~21k samples, great for Colab T4
    split_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42)
    train_hf = split_dataset["train"]
    val_hf = split_dataset["test"]

    print("Building VQA vocabulary...")
    questions = [sample["question"] for sample in train_hf]
    answers = [sample["multiple_choice_answer"] for sample in train_hf]

    vocab = Vocabulary(freq_threshold=3)
    vocab.build_vqa_vocabulary(questions, answers)
    vocab_size = len(vocab)
    print(f"Vocabulary built! Total unique tokens: {vocab_size}")

    # Stronger augmentations for training only
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Keep validation transform clean and deterministic
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Set is_vqa=True once you hook up your VQA dataset samples
    train_dataset = FlickrDataset(train_hf, vocab, transform=train_transform, is_vqa=True)
    val_dataset = FlickrDataset(val_hf, vocab, transform=val_transform, is_vqa=True)
    
    pad_idx = vocab.stoi["<PAD>"]
    collate_fn = CollateFn(pad_idx=pad_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # Initialize Model, Loss, Optimizer
    embed_dim = 512
    num_heads = 8
    hidden_dim = 2048
    num_layers = 4
    
    num_epochs = 20
    warmup_epochs = 2
    best_val_loss = float('inf')

    model = Bear(vocab_size, embed_dim, num_heads, hidden_dim, num_layers).to(device)
    
    # Ignore index set to -100 to omit question tokens and padding from loss computation
    criterion = nn.CrossEntropyLoss(
        ignore_index=-100,
        label_smoothing=0.05
    )

    cnn_params = [p for p in model.encoder.parameters() if p.requires_grad]
    transformer_params = list(model.decoder.parameters())

    optimizer = torch.optim.AdamW([
        {'params': cnn_params, 'lr': 5e-5},          # ResNet layer4 fine-tuning rate
        {'params': transformer_params, 'lr': 3e-4}   # Transformer decoder base rate
    ], weight_decay=1e-3)

    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    for epoch in range(num_epochs):
        # --- TRAINING ---
        model.train()
        total_train_loss = 0
        for batch_idx, (images, full_seqs, answer_start_idxs) in enumerate(train_loader):
            images = images.to(device)
            full_seqs = full_seqs.to(device)

            decoder_inputs = full_seqs[:, :-1]
            targets = full_seqs[:, 1:]

            # Construct targets mask (-100 for question/padding, active token indices for answers)
            targets_mask = targets.clone()
            for i, start_idx in enumerate(answer_start_idxs):
                ans_start_in_target = max(0, start_idx.item() - 1)
                targets_mask[i, :ans_start_in_target] = -100
                targets_mask[i, targets[i] == pad_idx] = -100

            optimizer.zero_grad()
            logits = model(images, decoder_inputs)
            loss = criterion(logits.reshape(-1, vocab_size), targets_mask.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- EVALUATION ---
        avg_val_loss = validate(model, val_loader, criterion, device, vocab_size, pad_idx)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[1]  # Index 1 corresponds to transformer_params
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
        
        # Save best checkpoint based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = "/content/drive/MyDrive/bear_vqa_checkpoint.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'vocab_size': vocab_size
            }, checkpoint_path)
            print(f"Model checkpoint saved successfully to {checkpoint_path}!")

if __name__ == "__main__":
    train()