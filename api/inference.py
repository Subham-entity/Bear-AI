import torch

def generate_caption(model, vocab, tensor_image, beam_width=3, max_length=30, device="cpu"):
    model.eval()
    with torch.no_grad():
        encoder_features = model.encoder(tensor_image.unsqueeze(0).to(device))
        
        # Start with the <START> token: tuple of (sequence_indices, score)
        start_token = vocab.stoi["<START>"]
        end_token = vocab.stoi["<END>"]
        
        beams = [([start_token], 0.0)]
        
        for _ in range(max_length):
            new_beams = []
            for seq, score in beams:
                if seq[-1] == end_token:
                    new_beams.append((seq, score))
                    continue
                
                captions_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
                logits = model.decoder(captions_tensor, encoder_features)
                
                # Get log probabilities of the next token
                log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
                topk_probs, topk_indices = torch.topk(log_probs, beam_width)
                
                for i in range(beam_width):
                    next_token = topk_indices[i].item()
                    next_score = topk_probs[i].item()
                    new_beams.append((seq + [next_token], score + next_score))
            
            # Sort beams by highest cumulative score and keep top 'beam_width'
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # Stop early if all beams end with <END>
            if all(seq[-1] == end_token for seq, _ in beams):
                break
                
        best_seq = beams[0][0]
        generated_words = [vocab.itos[idx] for idx in best_seq[1:] if idx != end_token]
        return " ".join(generated_words)