import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from datasets import load_dataset
from torch.optim.lr_scheduler import LambdaLR
from dataset.vocab import Vocabulary
from dataset.dataloader import MultimodalDataset, CollateFn
from models.bear import Bear


def validate(model, dataloader, criterion, device, vocab_size, pad_idx):
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for images, full_seqs, answer_start_idxs in dataloader:
            images = images.to(device)
            full_seqs = full_seqs.to(device)

            decoder_inputs = full_seqs[:, :-1]
            targets = full_seqs[:, 1:]

            # Mask prompt/question and padding tokens with -100[cite: 16]
            targets_mask = targets.clone()
            for i, start_idx in enumerate(answer_start_idxs):
                ans_start_in_target = max(0, start_idx.item() - 1)
                targets_mask[i, :ans_start_in_target] = -100
                targets_mask[i, targets[i] == pad_idx] = -100

            logits = model(images, decoder_inputs)
            loss = criterion(logits.reshape(-1, vocab_size), targets_mask.reshape(-1))
            total_val_loss += loss.item()

    return total_val_loss / len(dataloader)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load & split Flickr8k (Captioning)[cite: 8]
    print("Loading Flickr8k dataset...")
    flickr_hf = load_dataset("jxie/flickr8k", split="train")
    flickr_split = flickr_hf.train_test_split(test_size=0.1, seed=42)
    flickr_train_hf, flickr_val_hf = flickr_split["train"], flickr_split["test"]

    # 2. Load & split VQAv2-small (Visual Question Answering)[cite: 16]
    print("Loading VQA dataset...")
    vqa_hf = load_dataset("merve/vqav2-small", split="validation")
    vqa_split = vqa_hf.train_test_split(test_size=0.1, seed=42)
    vqa_train_hf, vqa_val_hf = vqa_split["train"], vqa_split["test"]

    # 3. Build unified vocabulary across both tasks[cite: 8, 12, 16]
    print("Building unified Captioning + VQA vocabulary...")
    all_texts = ["describe the image"]

    for sample in flickr_train_hf:
        for i in range(5):
            all_texts.append(sample[f"caption_{i}"])

    for sample in vqa_train_hf:
        all_texts.append(sample["question"])
        all_texts.append(sample["multiple_choice_answer"])

    vocab = Vocabulary(freq_threshold=3)
    vocab.build_vocabulary(all_texts)
    vocab_size = len(vocab)
    print(f"Unified vocabulary built! Total unique tokens: {vocab_size}")

    # 4. VQA-safe augmentations (no horizontal flip or heavy crop that breaks left/right & counts)
    # Spatial-safe augmentations for combined Captioning + VQA training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Keep validation transform clean and deterministic
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 5. Wrap datasets & combine for training
    flickr_train_ds = MultimodalDataset(flickr_train_hf, vocab, transform=train_transform, is_vqa=False)
    vqa_train_ds = MultimodalDataset(vqa_train_hf, vocab, transform=train_transform, is_vqa=True)
    combined_train_ds = ConcatDataset([flickr_train_ds, vqa_train_ds])

    flickr_val_ds = MultimodalDataset(flickr_val_hf, vocab, transform=val_transform, is_vqa=False)
    vqa_val_ds = MultimodalDataset(vqa_val_hf, vocab, transform=val_transform, is_vqa=True)

    pad_idx = vocab.stoi["<PAD>"]
    collate_fn = CollateFn(pad_idx=pad_idx)

    train_loader = DataLoader(
        combined_train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    flickr_val_loader = DataLoader(
        flickr_val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=2
    )
    vqa_val_loader = DataLoader(
        vqa_val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=2
    )

    # 6. Initialize Model, Loss, Optimizer, and Scheduler[cite: 16]
    embed_dim = 512
    num_heads = 8
    hidden_dim = 2048
    num_layers = 4

    num_epochs = 20
    warmup_epochs = 2
    best_val_loss = float("inf")

    model = Bear(vocab_size, embed_dim, num_heads, hidden_dim, num_layers).to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=-100,
        label_smoothing=0.05
    )

    cnn_params = [p for p in model.encoder.parameters() if p.requires_grad]
    transformer_params = list(model.decoder.parameters())

    optimizer = torch.optim.AdamW([
        {"params": cnn_params, "lr": 5e-5},
        {"params": transformer_params, "lr": 3e-4}
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
        total_train_loss = 0.0

        for images, full_seqs, answer_start_idxs in train_loader:
            images = images.to(device)
            full_seqs = full_seqs.to(device)

            decoder_inputs = full_seqs[:, :-1]
            targets = full_seqs[:, 1:]

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
        cap_val_loss = validate(model, flickr_val_loader, criterion, device, vocab_size, pad_idx)
        vqa_val_loss = validate(model, vqa_val_loader, criterion, device, vocab_size, pad_idx)
        avg_val_loss = (cap_val_loss + vqa_val_loss) / 2.0

        scheduler.step()
        current_lr = scheduler.get_last_lr()[1]
        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train: {avg_train_loss:.4f} | "
            f"Cap Val: {cap_val_loss:.4f} | "
            f"VQA Val: {vqa_val_loss:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Save best checkpoint based on combined validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = "/content/drive/MyDrive/bear_vqa_checkpoint.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "vocab_size": vocab_size
            }, checkpoint_path)
            print(f"Model checkpoint saved successfully to {checkpoint_path}!")


if __name__ == "__main__":
    train()