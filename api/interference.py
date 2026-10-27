import torch

def generate_caption(model, vocab, image, max_length=30, device="cpu"):
    model.eval()
    with torch.no_grad():
        # 1. Extract image features once -> shape: (1, 49, Embed_Dim)
        image = image.unsqueeze(0).to(device)
        encoder_features = model.encoder(image)

        # 2. Start with the <START> token
        token_indices = [vocab.stoi["<START>"]]

        for _ in range(max_length):
            # Prepare current input sequence tensor -> shape: (1, current_seq_len)
            captions_tensor = torch.tensor(token_indices, dtype=torch.long).unsqueeze(0).to(device)

            # Forward pass through decoder
            logits = model.decoder(captions_tensor, encoder_features)
            
            # Get the predicted token for the last time step
            predicted_token_id = logits[0, -1, :].argmax().item()

            # If model predicts <END>, stop generation
            if predicted_token_id == vocab.stoi["<END>"]:
                break

            token_indices.append(predicted_token_id)

        # 3. Convert token IDs back to words (ignoring <START>)
        generated_words = [vocab.itos[idx] for idx in token_indices[1:]]
        return " ".join(generated_words)